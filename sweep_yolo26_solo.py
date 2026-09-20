"""Solo eval of YOLO26m-seg trained with DEFAULT settings (no cls= hack),
to test whether YOLO26's native STAL/ProgLoss innovations already fix the
confidence-miscalibration bug we had to manually patch in YOLO11 (cls=1.5),
Mask R-CNN (cls=3.0), and RT-DETR (cls=2.0+60ep) -- see README.md. Mirrors
sweep_rtdetr_cls_solo.py's diagnostic + PQ-sweep pattern exactly, swapped
to YOLO26's API (same ultralytics YOLO() class as YOLO11/YOLO26-seg).
"""
import numpy as np
import torch
import pycocotools.mask as mu
from PIL import Image
from ultralytics import YOLO

from dataset import train_val_split, IMG_DIR, H, W
from train_refiner import RefinerUNet
from crop_dataset import square_bounds, CROP_SIZE
from predict_trained import to_rle
from predict_refined import refine_with_tta

YOLO26_CKPT = "kaggle_kernel_yolo26/output/yolo26m_best.pt"
REFINER_CKPT = "checkpoints/refiner_v5_best.pt"
FLOOR = 0.05
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
    yolo26 = YOLO(YOLO26_CKPT)

    rstate = torch.load(REFINER_CKPT, map_location=device)
    refiner = RefinerUNet(in_channels=rstate.get("in_channels", 1),
                           out_channels=rstate.get("out_channels", 1)).to(device)
    refiner.load_state_dict(rstate["model"])
    refiner.eval()

    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)

    per_image_cache = []
    all_scores = []
    localized_low_conf = []  # scores of well-localized (post-refine IoU>0.5) boxes

    with torch.no_grad():
        for i, e in enumerate(val_entries, 1):
            img_path = IMG_DIR / e["file_name"]
            gray = np.array(Image.open(img_path).convert("L"))

            out = yolo26.predict(source=str(img_path), imgsz=1280, conf=FLOOR, verbose=False)[0]
            cands_raw = []
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
                    cands_raw.append((float(scores[j]), coarse))

            gt = []
            for a in per_image.get(e["id"], []):
                rles = mu.frPyObjects(a["segmentation"], H, W)
                gt.append(to_rle(mu.decode(mu.merge(rles))))
            gt_dicts = [{"size": [H, W], "counts": r.encode()} for r in gt]

            refined = []
            for score, coarse in cands_raw:
                ref = refine_candidate(refiner, device, gray, coarse)
                all_scores.append(score)
                if gt_dicts:
                    pred_rle = {"size": [H, W], "counts": to_rle(ref).encode()}
                    iou = mu.iou([pred_rle], gt_dicts, [0] * len(gt_dicts))
                    if float(iou.max()) > 0.5:
                        localized_low_conf.append(score)
                refined.append((score, to_rle(ref)))

            per_image_cache.append((refined, gt))
            if i % 20 == 0:
                print(f"  cached {i}/{len(val_entries)}", flush=True)

    all_scores = np.array(all_scores)
    localized = np.array(localized_low_conf)
    print(f"\ntotal candidates: {len(all_scores)}, well-localized: {len(localized)}")
    print(f"well-localized score dist: min={localized.min():.3f} median={np.median(localized):.3f} "
          f"max={localized.max():.3f} mean={localized.mean():.3f}")
    print(f"well-localized below 0.5: {(localized < 0.5).sum()} ({100*(localized < 0.5).mean():.1f}%)")
    print("(compare: YOLO11 uncalibrated, RT-DETR uncalibrated had 58% below 0.5, median 0.031;")
    print(" YOLO11 cls=1.5 fix and RT-DETR cls=2.0 fix both needed to reach usable calibration)")

    print("\n=== confidence-threshold sweep (extended range, PQ) ===")
    print("=== references: YOLO11 cls=1.5 solo PQ 0.4252, RT-DETR cls=2.0 solo PQ 0.4296 ===\n")
    results = []
    for thresh in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                   0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        pq_num = pq_den = 0.0
        tp = 0
        for refined, gt in per_image_cache:
            filtered = [(s, r) for s, r in refined if s >= thresh]
            kept = paint_panoptic_rle(filtered)
            num, den, m_tp = pq_against_gt(kept, gt)
            pq_num += num
            pq_den += den
            tp += m_tp
        pq = pq_num / pq_den if pq_den else 0.0
        print(f"thresh={thresh}: PQ={pq:.4f} TP={tp}", flush=True)
        results.append((pq, thresh, tp))

    results.sort(reverse=True)
    print("\n=== top 5 ===")
    for pq, thresh, tp in results[:5]:
        print(f"PQ={pq:.4f}  thresh={thresh}  TP={tp}")


if __name__ == "__main__":
    main()
