"""Refine the refiner-confidence-gate finding: pure refiner confidence
doesn't discriminate real filaments from junk (sweep_refiner_gate.py showed
TP jumps 504->624 but PQ crashes 0.418->0.21 -- the refiner is overconfident
on out-of-distribution candidates). But the earlier diagnostic
(diag_missed_filaments.py) showed missed filaments DO have measurably lower
local contrast than caught ones (median 12.4 vs 16.5 gray levels) even
though the refiner can still segment them well when given the right box.

This tries a narrower "rescue" pass: keep everything YOLO scores >=0.35 (the
deployed baseline, untouched), and ADD BACK candidates YOLO scored in
[0.02, 0.35) only if the refined mask's local contrast clears a threshold --
a real, measured filament property, not the refiner's own (unreliable)
confidence. Single new axis (contrast threshold) swept against a cache built
once, same validated methodology as every other sweep this session.
"""
import numpy as np
import torch
import pycocotools.mask as mu
from PIL import Image
from ultralytics import YOLO

from dataset import train_val_split, IMG_DIR, H, W
from train_refiner import RefinerUNet
from crop_dataset import square_bounds, CROP_SIZE
from predict_refined import refine_with_tta
from predict_trained import to_rle

YOLO_CKPT = "kaggle_kernel_train_m/output/yolo11m_best.pt"
REFINER_CKPT = "checkpoints/refiner_v5_best.pt"
YOLO_FLOOR = 0.02
DEPLOYED_THRESH = 0.35
MIN_AREA = 20


def local_contrast(mask, gray):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 0.0
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    pad = 15
    by0, by1 = max(0, y0 - pad), min(H, y1 + pad)
    bx0, bx1 = max(0, x0 - pad), min(W, x1 + pad)
    region = gray[by0:by1, bx0:bx1].astype(np.float32)
    region_mask = mask[by0:by1, bx0:bx1].astype(bool)
    bg_pixels = region[~region_mask]
    inside_mean = gray[mask.astype(bool)].mean()
    bg_mean = bg_pixels.mean() if bg_pixels.size else inside_mean
    return float(bg_mean - inside_mean)


def paint_panoptic(candidates, min_area=MIN_AREA):
    candidates = sorted(candidates, key=lambda x: -x[0])
    claimed = np.zeros((H, W), dtype=np.uint8)
    kept = []
    for score, binary in candidates:
        remaining = binary & (1 - claimed)
        if int(remaining.sum()) < min_area:
            continue
        claimed |= remaining
        kept.append(remaining)
    return kept


def main():
    device = torch.device("cuda")
    yolo = YOLO(YOLO_CKPT)
    rstate = torch.load(REFINER_CKPT, map_location=device)
    refiner = RefinerUNet(in_channels=rstate.get("in_channels", 1),
                           out_channels=rstate.get("out_channels", 1)).to(device)
    refiner.load_state_dict(rstate["model"])
    refiner.eval()

    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)

    cache = []
    for i, e in enumerate(val_entries, 1):
        img_path = IMG_DIR / e["file_name"]
        gray = np.array(Image.open(img_path).convert("L"))
        out = yolo.predict(source=str(img_path), imgsz=1280, conf=YOLO_FLOOR, verbose=False)[0]

        deployed = []  # (score, mask) for yolo_score >= DEPLOYED_THRESH
        rescue_pool = []  # (score, mask, contrast) for yolo_score in [YOLO_FLOOR, DEPLOYED_THRESH)

        if out.boxes is not None and len(out.boxes) > 0:
            boxes = out.boxes.xyxy.cpu().numpy()
            yolo_scores = out.boxes.conf.cpu().numpy()
            for j in range(len(boxes)):
                x0, y0, x1, y1 = boxes[j]
                x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
                if x1 <= x0 or y1 <= y0:
                    continue
                coarse = np.zeros((H, W), dtype=np.uint8)
                coarse[y0:y1, x0:x1] = 1
                cx0, cy0, cx1, cy1 = square_bounds(coarse.astype(bool))
                crop = np.array(Image.fromarray(gray[cy0:cy1, cx0:cx1]).resize(
                    (CROP_SIZE, CROP_SIZE), Image.BILINEAR)).astype(np.float32) / 255.0
                with torch.no_grad():
                    prob = refine_with_tta(refiner, device, np.stack([crop]))
                side = cy1 - cy0
                prob_full = np.array(Image.fromarray((prob * 255).astype(np.uint8)).resize(
                    (side, side), Image.BILINEAR)).astype(np.float32) / 255.0
                refined_crop = (prob_full > 0.5).astype(np.uint8)
                full_mask = np.zeros((H, W), dtype=np.uint8)
                full_mask[cy0:cy1, cx0:cx1] = refined_crop
                if full_mask.sum() == 0:
                    continue

                score = float(yolo_scores[j])
                if score >= DEPLOYED_THRESH:
                    deployed.append((score, full_mask))
                else:
                    contrast = local_contrast(full_mask, gray)
                    rescue_pool.append((score, full_mask, contrast))

        gt = []
        for a in per_image.get(e["id"], []):
            rles = mu.frPyObjects(a["segmentation"], H, W)
            gt.append(to_rle(mu.decode(mu.merge(rles))))

        cache.append((deployed, rescue_pool, gt))
        if i % 10 == 0:
            avg_rescue = sum(len(c[1]) for c in cache) / i
            print(f"  cached {i}/{len(val_entries)} (avg {avg_rescue:.1f} rescue-pool "
                  f"candidates/image)", flush=True)

    print("cache built, sweeping contrast rescue threshold...\n", flush=True)

    def pq_for(contrast_thresh):
        pq_num = pq_den = 0.0
        tp = 0
        n_rescued = 0
        for deployed, rescue_pool, gt in cache:
            rescued = [(s, m) for s, m, c in rescue_pool if c >= contrast_thresh]
            n_rescued += len(rescued)
            kept = paint_panoptic(deployed + rescued)
            pred = [{"size": [H, W], "counts": to_rle(m).encode()} for m in kept]
            gt_rle = [{"size": [H, W], "counts": r.encode()} for r in gt]
            if pred and gt_rle:
                iou = mu.iou(pred, gt_rle, [0] * len(gt_rle))
                best_per_gt = iou.max(axis=0)
            else:
                best_per_gt = np.zeros(len(gt_rle))
            m_tp = int((best_per_gt > 0.5).sum())
            n_fp = len(pred) - m_tp
            n_fn = len(gt_rle) - m_tp
            pq_num += float(best_per_gt[best_per_gt > 0.5].sum())
            pq_den += m_tp + 0.5 * n_fp + 0.5 * n_fn
            tp += m_tp
        return (pq_num / pq_den if pq_den else 0.0), tp, n_rescued

    baseline_pq, baseline_tp, _ = pq_for(float("inf"))  # no rescue at all = current deployed baseline
    print(f"baseline (no rescue, yolo_conf>=0.35 only): PQ={baseline_pq:.4f} TP={baseline_tp}\n")

    results = []
    for thresh in [5, 8, 10, 12, 15, 18, 20, 25, 30]:
        pq, tp, n_rescued = pq_for(thresh)
        print(f"contrast>={thresh}: PQ={pq:.4f} TP={tp} (+{tp-baseline_tp} vs baseline, "
              f"avg {n_rescued/len(val_entries):.2f} rescued/image)", flush=True)
        results.append((pq, thresh, tp))

    results.sort(reverse=True)
    print("\n=== top 5 ===")
    for pq, thresh, tp in results[:5]:
        print(f"PQ={pq:.4f}  contrast_thresh={thresh}  TP={tp}")


if __name__ == "__main__":
    main()
