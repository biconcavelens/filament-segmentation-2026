"""Untested combination: YOLO11(cls=1.5) + RT-DETR(cls=2.0), WITHOUT Mask
R-CNN. Every RT-DETR ensemble test so far added it as a 3rd member ON TOP
of the validated Mask R-CNN + YOLO11 pipeline (2-way real 0.39; 3-way with
RT-DETR ties real 0.39). But RT-DETR solo (0.4296) is now the STRONGEST
solo detector of the three, beating YOLO11 (0.4252) and Mask R-CNN
(0.4152) -- worth testing whether replacing the weakest solo detector
(Mask R-CNN) with nothing, keeping just the two strongest, beats keeping
all three. Structurally, YOLO (CNN, anchor-based) and RT-DETR
(transformer, DETR-style) are different enough to plausibly still give
real ensemble diversity without Mask R-CNN's two-stage RPN architecture.
"""
import numpy as np
import torch
import pycocotools.mask as mu
from PIL import Image
from ultralytics import YOLO, RTDETR
from sklearn.isotonic import IsotonicRegression

from dataset import train_val_split, IMG_DIR, H, W
from train_refiner import RefinerUNet
from crop_dataset import square_bounds, CROP_SIZE
from predict_trained import to_rle
from predict_refined import refine_with_tta

YOLO_CKPT = "kaggle_dataset_upload/yolo11m_cls_best.pt"
RTDETR_CKPT = "kaggle_kernel_rtdetr_cls/output/rtdetr_cls_best.pt"
REFINER_CKPT = "checkpoints/refiner_v5_best.pt"
FLOOR_B, FLOOR_C = 0.05, 0.05
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
    candidates = sorted(candidates, key=lambda x: -x[0])
    accepted, accepted_masks = [], []
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


def _yolo_style_candidates(model, img_path, floor):
    out = model.predict(source=str(img_path), imgsz=1280, conf=floor, verbose=False)[0]
    cands = []
    if out.boxes is not None and len(out.boxes) > 0:
        boxes = out.boxes.xyxy.cpu().numpy()
        scores = out.boxes.conf.cpu().numpy()
        for j in range(len(boxes)):
            x0, y0, x1, y1 = boxes[j]
            x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
            if x1 <= x0 or y1 <= y0:
                continue
            coarse = np.zeros((H, W), dtype=np.uint8)
            coarse[y0:y1, x0:x1] = 1
            cands.append((float(scores[j]), coarse))
    return cands


def main():
    device = torch.device("cuda")
    yolo = YOLO(YOLO_CKPT)
    rtdetr = RTDETR(RTDETR_CKPT)

    rstate = torch.load(REFINER_CKPT, map_location=device)
    refiner = RefinerUNet(in_channels=rstate.get("in_channels", 1),
                           out_channels=rstate.get("out_channels", 1)).to(device)
    refiner.load_state_dict(rstate["model"])
    refiner.eval()

    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)

    per_image_cache = []
    scoresB_all, labelsB_all = [], []
    scoresC_all, labelsC_all = [], []

    with torch.no_grad():
        for i, e in enumerate(val_entries, 1):
            img_path = IMG_DIR / e["file_name"]
            gray = np.array(Image.open(img_path).convert("L"))

            candsB_raw = _yolo_style_candidates(yolo, img_path, FLOOR_B)
            candsC_raw = _yolo_style_candidates(rtdetr, img_path, FLOOR_C)

            gt = []
            for a in per_image.get(e["id"], []):
                rles = mu.frPyObjects(a["segmentation"], H, W)
                gt.append(to_rle(mu.decode(mu.merge(rles))))

            refinedB, refinedC = [], []
            for score, coarse in candsB_raw:
                ref = refine_candidate(refiner, device, gray, coarse)
                label = 1 if best_iou_against_gt(ref, [{"size": [H, W], "counts": r.encode()} for r in gt]) > 0.5 else 0
                refinedB.append((score, to_rle(ref)))
                scoresB_all.append(score)
                labelsB_all.append(label)
            for score, coarse in candsC_raw:
                ref = refine_candidate(refiner, device, gray, coarse)
                label = 1 if best_iou_against_gt(ref, [{"size": [H, W], "counts": r.encode()} for r in gt]) > 0.5 else 0
                refinedC.append((score, to_rle(ref)))
                scoresC_all.append(score)
                labelsC_all.append(label)

            per_image_cache.append((refinedB, refinedC, gt))
            if i % 20 == 0:
                print(f"  cached {i}/{len(val_entries)}", flush=True)

    print(f"yolo11 candidates: {len(scoresB_all)} ({sum(labelsB_all)} TP)", flush=True)
    print(f"rtdetr candidates: {len(scoresC_all)} ({sum(labelsC_all)} TP)", flush=True)

    calB = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calB.fit(scoresB_all, labelsB_all)
    calC = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calC.fit(scoresC_all, labelsC_all)

    calibrated_cache = []
    for refinedB, refinedC, gt in per_image_cache:
        pooled = []
        if refinedB:
            cs = calB.predict([s for s, _ in refinedB])
            pooled.extend((float(c), r) for c, (_, r) in zip(cs, refinedB))
        if refinedC:
            cs = calC.predict([s for s, _ in refinedC])
            pooled.extend((float(c), r) for c, (_, r) in zip(cs, refinedC))
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

    print("\n=== solo: YOLO11 0.4252, RT-DETR 0.4296, Mask R-CNN 0.4152 (excluded here) ===")
    print("=== reference: Mask R-CNN+YOLO11 2-way best PQ=0.4407 (real 0.39) ===\n")
    results = []
    for dedup_iou in [0.03, 0.05, 0.08, 0.1, 0.15]:
        for accept_thresh in [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
            pq, tp = pq_for(dedup_iou, accept_thresh)
            print(f"dedup_iou={dedup_iou} accept={accept_thresh}: PQ={pq:.4f} TP={tp}", flush=True)
            results.append((pq, dedup_iou, accept_thresh, tp))

    results.sort(reverse=True)
    print("\n=== top 5 ===")
    for pq, dedup_iou, accept_thresh, tp in results[:5]:
        print(f"PQ={pq:.4f}  dedup_iou={dedup_iou}  accept={accept_thresh}  TP={tp}")


if __name__ == "__main__":
    main()
