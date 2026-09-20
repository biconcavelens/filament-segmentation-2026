"""Four-source ensemble: Mask R-CNN + YOLO@1280 + YOLO@2048 (native res) +
RT-DETR(cls=2.0/60ep), all pooled via the same true-NMS-dedup mechanism.

Two independent local wins this session did NOT transfer to the real
leaderboard on their own: RT-DETR recalibration (+0.0011 local, tied real)
and multi-scale YOLO TTA (+0.0199 local, largest of the session, but STILL
tied real at 0.38 solo). Every genuine real-score improvement so far has
come from ensembling multiple sources, not tuning any single one -- this
tests whether combining both non-transferring wins into the existing
validated ensemble crosses the leaderboard's rounding threshold where
neither did alone.
"""
import numpy as np
import torch
import pycocotools.mask as mu
from PIL import Image
from ultralytics import YOLO, RTDETR
from sklearn.isotonic import IsotonicRegression

from dataset import train_val_split, IMG_DIR, H, W
from train import build_from_checkpoint
from train_refiner import RefinerUNet
from crop_dataset import square_bounds, CROP_SIZE
from predict_trained import to_rle
from predict_refined import refine_with_tta

MASKRCNN_CKPT = "kaggle_kernel_maskrcnn_cls/output/checkpoints/maskrcnn_cls_epoch5.pt"
YOLO_CKPT = "kaggle_dataset_upload/yolo11m_cls_best.pt"
RTDETR_CKPT = "kaggle_kernel_rtdetr_cls/output/rtdetr_cls_best.pt"
REFINER_CKPT = "checkpoints/refiner_v5_best.pt"
FLOOR_A, FLOOR_B1280, FLOOR_B2048, FLOOR_C = 0.3, 0.05, 0.05, 0.05
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


def _yolo_style_candidates(model, img_path, imgsz, floor):
    out = model.predict(source=str(img_path), imgsz=imgsz, conf=floor, verbose=False)[0]
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
    stateA = torch.load(MASKRCNN_CKPT, map_location=device)
    detA = build_from_checkpoint(stateA, num_classes=2).to(device)
    detA.load_state_dict(stateA["model"])
    detA.roi_heads.nms_thresh = 0.30
    detA.eval()

    yolo = YOLO(YOLO_CKPT)
    rtdetr = RTDETR(RTDETR_CKPT)

    rstate = torch.load(REFINER_CKPT, map_location=device)
    refiner = RefinerUNet(in_channels=rstate.get("in_channels", 1),
                           out_channels=rstate.get("out_channels", 1)).to(device)
    refiner.load_state_dict(rstate["model"])
    refiner.eval()

    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)

    per_image_cache = []
    scores_all = {"A": [], "B1280": [], "B2048": [], "C": []}
    labels_all = {"A": [], "B1280": [], "B2048": [], "C": []}

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

            candsB1280_raw = _yolo_style_candidates(yolo, img_path, 1280, FLOOR_B1280)
            candsB2048_raw = _yolo_style_candidates(yolo, img_path, 2048, FLOOR_B2048)
            candsC_raw = _yolo_style_candidates(rtdetr, img_path, 1280, FLOOR_C)

            gt = []
            for a in per_image.get(e["id"], []):
                rles = mu.frPyObjects(a["segmentation"], H, W)
                gt.append(to_rle(mu.decode(mu.merge(rles))))
            gt_dicts = [{"size": [H, W], "counts": r.encode()} for r in gt]

            per_source = {}
            for key, cands in [("A", candsA_raw), ("B1280", candsB1280_raw),
                                ("B2048", candsB2048_raw), ("C", candsC_raw)]:
                refined = []
                for score, coarse in cands:
                    ref = refine_candidate(refiner, device, gray, coarse)
                    if ref.sum() == 0:
                        continue
                    label = 1 if best_iou_against_gt(ref, gt_dicts) > 0.5 else 0
                    refined.append((score, to_rle(ref)))
                    scores_all[key].append(score)
                    labels_all[key].append(label)
                per_source[key] = refined

            per_image_cache.append((per_source, gt))
            if i % 20 == 0:
                print(f"  cached {i}/{len(val_entries)}", flush=True)

    for key in scores_all:
        print(f"{key}: {len(scores_all[key])} candidates ({sum(labels_all[key])} TP)", flush=True)

    cals = {}
    for key in scores_all:
        cal = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        cal.fit(scores_all[key], labels_all[key])
        cals[key] = cal

    calibrated_cache = []
    for per_source, gt in per_image_cache:
        pooled = []
        for key, refined in per_source.items():
            if refined:
                cs = cals[key].predict([s for s, _ in refined])
                pooled.extend((float(c), r) for c, (_, r) in zip(cs, refined))
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

    print("\n=== references: 2-way ens 0.4407 (real 0.39), 3-way+RTDETR 0.4418 (real 0.39, tied)")
    print("=== solo multiscale YOLO 0.4451 (real 0.38, tied) ===\n")
    results = []
    for dedup_iou in [0.03, 0.05, 0.08, 0.1]:
        for accept_thresh in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
            pq, tp = pq_for(dedup_iou, accept_thresh)
            print(f"dedup_iou={dedup_iou} accept={accept_thresh}: PQ={pq:.4f} TP={tp}", flush=True)
            results.append((pq, dedup_iou, accept_thresh, tp))

    results.sort(reverse=True)
    print("\n=== top 5 ===")
    for pq, dedup_iou, accept_thresh, tp in results[:5]:
        print(f"PQ={pq:.4f}  dedup_iou={dedup_iou}  accept={accept_thresh}  TP={tp}")


if __name__ == "__main__":
    main()
