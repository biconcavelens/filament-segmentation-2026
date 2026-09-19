"""Fifth ensemble attempt, and the most surgical one yet. Diagnosis so far:
pooling raw candidates from two architectures means the same true filament
often gets several overlapping proposals; panoptic-paint's greedy pixel-
claiming lets the loser's leftover fragment show up as an extra near-miss/
spurious prediction instead of disappearing cleanly. Every previous attempt
(hand-tuned thresholds, grid search, calibrated via isotonic regression x2)
pooled candidates and let paint_panoptic sort out overlaps implicitly. This
is the first attempt to actually fix that: explicit cross-detector NMS
before painting -- sort all calibrated candidates by score, greedily
accept, and DISCARD (not fragment) anything that overlaps an already-
accepted candidate above an IoU threshold.

Caches candidates as RLE strings, not dense arrays -- an earlier local
session crashed the whole machine on this same mistake (storing dense
2048x2048 masks for both detectors' candidates across all 116 val images
exhausted this machine's 15GB RAM). Dense arrays are only ever materialized
transiently, per image, inside dedup_nms/paint_panoptic.
"""
import numpy as np
import torch
import pycocotools.mask as mu
from PIL import Image
from ultralytics import YOLO
from sklearn.isotonic import IsotonicRegression

from dataset import train_val_split, IMG_DIR, H, W
from train import build_from_checkpoint
from train_refiner import RefinerUNet
from crop_dataset import square_bounds, CROP_SIZE
from predict_trained import to_rle
from predict_refined import refine_with_tta

MASKRCNN_CKPT = "kaggle_kernel_maskrcnn_cls/output/checkpoints/maskrcnn_cls_epoch5.pt"
YOLO_CKPT = "kaggle_dataset_upload/yolo11m_cls_best.pt"
REFINER_CKPT = "checkpoints/refiner_v5_best.pt"
FLOOR_A, FLOOR_B = 0.5, 0.15
MIN_AREA = 20


@torch.no_grad()
def refine_candidate(refiner, device, gray, coarse):
    x0, y0, x1, y1 = square_bounds(coarse.astype(bool))
    crop = np.array(Image.fromarray(gray[y0:y1, x0:x1]).resize(
        (CROP_SIZE, CROP_SIZE), Image.BILINEAR)).astype(np.float32) / 255.0
    prob = refine_with_tta(refiner, device, np.stack([crop]))
    side = y1 - y0
    prob_full = np.array(Image.fromarray((prob * 255).astype(np.uint8)).resize(
        (side, side), Image.BILINEAR)).astype(np.float32) / 255.0
    refined_crop = (prob_full > 0.5).astype(np.uint8)
    full_mask = np.zeros((H, W), dtype=np.uint8)
    full_mask[y0:y1, x0:x1] = refined_crop
    return full_mask


def best_iou_against_gt(mask, gt_rles):
    if not gt_rles:
        return 0.0
    pred_rle = {"size": [H, W], "counts": to_rle(mask).encode()}
    iou = mu.iou([pred_rle], gt_rles, [0] * len(gt_rles))
    return float(iou.max())


def dedup_nms_rle(candidates, iou_thresh):
    """candidates: list of (score, rle_string). True NMS: sort by score,
    greedily accept; discard (don't fragment) anything overlapping an
    already-accepted candidate above iou_thresh. Decodes to dense only
    transiently, for this one image's candidates."""
    candidates = sorted(candidates, key=lambda x: -x[0])
    accepted = []
    accepted_masks = []
    for score, rle in candidates:
        mask = mu.decode({"size": [H, W], "counts": rle.encode()})
        is_dup = False
        for amask in accepted_masks:
            inter = np.logical_and(mask, amask).sum()
            if inter == 0:
                continue
            union = np.logical_or(mask, amask).sum()
            if inter / union > iou_thresh:
                is_dup = True
                break
        if not is_dup:
            accepted.append((score, rle))
            accepted_masks.append(mask)
    return accepted


def paint_panoptic_rle(candidates, min_area=MIN_AREA):
    candidates = sorted(candidates, key=lambda x: -x[0])
    claimed = np.zeros((H, W), dtype=np.uint8)
    kept = []
    for score, rle in candidates:
        binary = mu.decode({"size": [H, W], "counts": rle.encode()})
        remaining = binary & (1 - claimed)
        area = int(remaining.sum())
        if area < min_area:
            continue
        claimed |= remaining
        kept.append(to_rle(remaining))
    return kept


def pq_against_gt(kept_rles, gt_rles):
    pred = [{"size": [H, W], "counts": r.encode()} for r in kept_rles]
    gt_dicts = [{"size": [H, W], "counts": r.encode()} for r in gt_rles]
    if pred and gt_dicts:
        iou = mu.iou(pred, gt_dicts, [0] * len(gt_dicts))
        best_per_gt = iou.max(axis=0)
    else:
        best_per_gt = np.zeros(len(gt_rles))
    m_tp = int((best_per_gt > 0.5).sum())
    n_fp = len(pred) - m_tp
    n_fn = len(gt_rles) - m_tp
    num = float(best_per_gt[best_per_gt > 0.5].sum())
    den = m_tp + 0.5 * n_fp + 0.5 * n_fn
    return num, den, m_tp


def main():
    device = torch.device("cuda")
    stateA = torch.load(MASKRCNN_CKPT, map_location=device)
    detA = build_from_checkpoint(stateA, num_classes=2).to(device)
    detA.load_state_dict(stateA["model"])
    detA.roi_heads.nms_thresh = 0.30
    detA.eval()

    yolo = YOLO(YOLO_CKPT)

    rstate = torch.load(REFINER_CKPT, map_location=device)
    refiner = RefinerUNet(in_channels=rstate.get("in_channels", 1),
                           out_channels=rstate.get("out_channels", 1)).to(device)
    refiner.load_state_dict(rstate["model"])
    refiner.eval()

    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)

    per_image_cache = []
    scoresA_all, labelsA_all = [], []
    scoresB_all, labelsB_all = [], []

    with torch.no_grad():
        for i, e in enumerate(val_entries, 1):
            img_path = IMG_DIR / e["file_name"]
            gray = np.array(Image.open(img_path).convert("L"))
            rgb = np.array(Image.open(img_path).convert("RGB"))
            img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

            outA = detA([img_t.to(device)])[0]
            scoresA = outA["scores"].cpu().numpy()
            masksA = outA["masks"].cpu().numpy()
            candsA_raw = [(float(scoresA[j]), (masksA[j, 0] > 0.5).astype(np.uint8))
                          for j in range(len(scoresA))
                          if scoresA[j] >= FLOOR_A and (masksA[j, 0] > 0.5).sum() > 0]

            outB = yolo.predict(source=str(img_path), imgsz=1280, conf=FLOOR_B, verbose=False)[0]
            candsB_raw = []
            if outB.boxes is not None and len(outB.boxes) > 0:
                boxes = outB.boxes.xyxy.cpu().numpy()
                scoresB = outB.boxes.conf.cpu().numpy()
                for j in range(len(boxes)):
                    x0, y0, x1, y1 = boxes[j]
                    x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
                    if x1 <= x0 or y1 <= y0:
                        continue
                    coarse = np.zeros((H, W), dtype=np.uint8)
                    coarse[y0:y1, x0:x1] = 1
                    candsB_raw.append((float(scoresB[j]), coarse))

            gt = []
            for a in per_image.get(e["id"], []):
                rles = mu.frPyObjects(a["segmentation"], H, W)
                gt.append(to_rle(mu.decode(mu.merge(rles))))

            refinedA, refinedB = [], []
            for score, coarse in candsA_raw:
                ref = refine_candidate(refiner, device, gray, coarse)
                label = 1 if best_iou_against_gt(ref, [{"size": [H, W], "counts": r.encode()} for r in gt]) > 0.5 else 0
                refinedA.append((score, to_rle(ref)))
                scoresA_all.append(score)
                labelsA_all.append(label)
            for score, coarse in candsB_raw:
                ref = refine_candidate(refiner, device, gray, coarse)
                label = 1 if best_iou_against_gt(ref, [{"size": [H, W], "counts": r.encode()} for r in gt]) > 0.5 else 0
                refinedB.append((score, to_rle(ref)))
                scoresB_all.append(score)
                labelsB_all.append(label)

            per_image_cache.append((refinedA, refinedB, gt))
            if i % 20 == 0:
                print(f"  cached {i}/{len(val_entries)}", flush=True)

    print(f"maskrcnn candidates: {len(scoresA_all)} ({sum(labelsA_all)} TP)", flush=True)
    print(f"yolo candidates: {len(scoresB_all)} ({sum(labelsB_all)} TP)", flush=True)

    calA = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calA.fit(scoresA_all, labelsA_all)
    calB = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calB.fit(scoresB_all, labelsB_all)

    calibrated_cache = []
    for refinedA, refinedB, gt in per_image_cache:
        pooled = []
        if refinedA:
            cs = calA.predict([s for s, _ in refinedA])
            pooled.extend((float(c), r) for c, (_, r) in zip(cs, refinedA))
        if refinedB:
            cs = calB.predict([s for s, _ in refinedB])
            pooled.extend((float(c), r) for c, (_, r) in zip(cs, refinedB))
        calibrated_cache.append((pooled, gt))
    del per_image_cache

    def pq_for(dedup_iou, accept_thresh):
        pq_num = pq_den = 0.0
        tp = 0
        for pooled, gt in calibrated_cache:
            filtered = [(s, r) for s, r in pooled if s >= accept_thresh]
            deduped = dedup_nms_rle(filtered, dedup_iou)
            kept = paint_panoptic_rle(deduped)
            num, den, m_tp = pq_against_gt(kept, gt)
            pq_num += num
            pq_den += den
            tp += m_tp
        return (pq_num / pq_den if pq_den else 0.0), tp

    print("\n=== solo baselines: YOLO cls-fixed 0.4252, Mask R-CNN cls-fixed 0.4152 ===\n")
    print("=== fine accept_thresh sweep at dedup_iou=0.05 (established plateau) ===\n")
    results = []
    for dedup_iou in [0.05]:
        for accept_thresh in [0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48]:
            pq, tp = pq_for(dedup_iou, accept_thresh)
            print(f"dedup_iou={dedup_iou} accept={accept_thresh}: PQ={pq:.4f} TP={tp}", flush=True)
            results.append((pq, dedup_iou, accept_thresh, tp))

    results.sort(reverse=True)
    print("\n=== top 5 ===")
    for pq, dedup_iou, accept_thresh, tp in results[:5]:
        print(f"PQ={pq:.4f}  dedup_iou={dedup_iou}  accept={accept_thresh}  TP={tp}")


if __name__ == "__main__":
    main()
