"""Third attempt at rescuing YOLO's low-confidence-but-well-localized
candidates (diag_missed_filaments.py's root-cause finding). Two single-
feature rescue filters already failed to generalize: raw refiner confidence
(overconfident on junk) and local contrast (recovered only +4 TP, still
net-negative on PQ). This tries combining several weak signals -- raw YOLO
score, local contrast, area, elongation, thickness -- via a small logistic
regression classifier instead of a single threshold, since weak signals
sometimes separate jointly what none can alone.

Fit on a genuinely separate pool this time: candidates from a SUBSET OF
TRAIN images (with real GT labels), not the val set itself -- the earlier
isotonic-calibration attempts were fit and evaluated on the same val
candidates, a methodological shortcut worth avoiding for a change-of-
architecture like this. Rescue decisions and final PQ are evaluated on the
untouched val split.

Caches candidates as compact RLE strings (not dense 2048x2048 arrays) --
storing dense masks for thousands of candidates across 366 images exhausted
this machine's 15GB RAM (only 4.9GB free) on the first attempt. Dense
arrays are only ever materialized transiently, per image, inside
paint_panoptic.
"""
import random

import numpy as np
import torch
import pycocotools.mask as mu
from PIL import Image
from ultralytics import YOLO
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from dataset import train_val_split, IMG_DIR, H, W
from train_refiner import RefinerUNet
from crop_dataset import square_bounds, CROP_SIZE
from predict_trained import to_rle
from predict_refined import refine_with_tta

YOLO_CKPT = "kaggle_dataset_upload/yolo11m_cls_best.pt"
REFINER_CKPT = "checkpoints/refiner_v5_best.pt"
DEPLOYED_THRESH = 0.33
YOLO_FLOOR = 0.02
MIN_AREA = 20
N_TRAIN_IMAGES_FOR_FIT = 250


def local_contrast_and_shape(mask, gray):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    h, w = y1 - y0, x1 - x0
    long_side, short_side = max(h, w), max(1, min(h, w))
    elongation = long_side / short_side
    area = len(ys)
    thickness = area / max(1, long_side)

    pad = 15
    by0, by1 = max(0, y0 - pad), min(H, y1 + pad)
    bx0, bx1 = max(0, x0 - pad), min(W, x1 + pad)
    region = gray[by0:by1, bx0:bx1].astype(np.float32)
    region_mask = mask[by0:by1, bx0:bx1].astype(bool)
    bg_pixels = region[~region_mask]
    inside_mean = gray[mask.astype(bool)].mean()
    bg_mean = bg_pixels.mean() if bg_pixels.size else inside_mean
    contrast = bg_mean - inside_mean
    return dict(area=area, elongation=elongation, thickness=thickness, contrast=contrast)


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
    """candidates: list of (score, rle_string). Decodes to dense only here,
    transiently, for one image's worth of candidates at a time."""
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


def gather_candidates(entries, yolo, refiner, device, per_image, want_gt=True):
    """Returns list of (deployed_cands, rescue_pool, gt) per image. All
    masks stored as RLE strings, not dense arrays, to keep memory bounded
    across hundreds of images' candidates."""
    out = []
    for i, e in enumerate(entries, 1):
        img_path = IMG_DIR / e["file_name"]
        gray = np.array(Image.open(img_path).convert("L"))
        out_pred = yolo.predict(source=str(img_path), imgsz=1280, conf=YOLO_FLOOR, verbose=False)[0]

        deployed, rescue_pool = [], []
        if out_pred.boxes is not None and len(out_pred.boxes) > 0:
            boxes = out_pred.boxes.xyxy.cpu().numpy()
            scores = out_pred.boxes.conf.cpu().numpy()
            for j in range(len(boxes)):
                x0, y0, x1, y1 = boxes[j]
                x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
                if x1 <= x0 or y1 <= y0:
                    continue
                coarse = np.zeros((H, W), dtype=np.uint8)
                coarse[y0:y1, x0:x1] = 1
                full_mask = refine_candidate(refiner, device, gray, coarse)
                if full_mask.sum() == 0:
                    continue
                score = float(scores[j])
                if score >= DEPLOYED_THRESH:
                    deployed.append((score, to_rle(full_mask)))
                else:
                    feats = local_contrast_and_shape(full_mask, gray)
                    if feats is not None:
                        rescue_pool.append((score, to_rle(full_mask), feats))

        gt = []
        if want_gt:
            for a in per_image.get(e["id"], []):
                rles = mu.frPyObjects(a["segmentation"], H, W)
                gt.append(to_rle(mu.decode(mu.merge(rles))))

        out.append((deployed, rescue_pool, gt))
        if i % 25 == 0:
            print(f"  {i}/{len(entries)}", flush=True)
    return out


def best_iou_against_rle(rle, gt_rles):
    if not gt_rles:
        return 0.0
    pred_rle = {"size": [H, W], "counts": rle.encode()}
    gt_rle = [{"size": [H, W], "counts": r.encode()} for r in gt_rles]
    iou = mu.iou([pred_rle], gt_rle, [0] * len(gt_rle))
    return float(iou.max())


def pq_against_gt(kept_rles, gt_rles):
    pred = [{"size": [H, W], "counts": r.encode()} for r in kept_rles]
    gt_rle = [{"size": [H, W], "counts": r.encode()} for r in gt_rles]
    if pred and gt_rle:
        iou = mu.iou(pred, gt_rle, [0] * len(gt_rle))
        best_per_gt = iou.max(axis=0)
    else:
        best_per_gt = np.zeros(len(gt_rle))
    m_tp = int((best_per_gt > 0.5).sum())
    n_fp = len(pred) - m_tp
    n_fn = len(gt_rle) - m_tp
    num = float(best_per_gt[best_per_gt > 0.5].sum())
    den = m_tp + 0.5 * n_fp + 0.5 * n_fn
    return num, den, m_tp


def main():
    device = torch.device("cuda")
    yolo = YOLO(YOLO_CKPT)
    rstate = torch.load(REFINER_CKPT, map_location=device)
    refiner = RefinerUNet(in_channels=rstate.get("in_channels", 1),
                           out_channels=rstate.get("out_channels", 1)).to(device)
    refiner.load_state_dict(rstate["model"])
    refiner.eval()

    train_entries, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)
    rng = random.Random(0)
    fit_entries = rng.sample(train_entries, min(N_TRAIN_IMAGES_FOR_FIT, len(train_entries)))

    print(f"gathering fit pool from {len(fit_entries)} train images...", flush=True)
    fit_cache = gather_candidates(fit_entries, yolo, refiner, device, per_image)

    X_fit, y_fit = [], []
    for _, rescue_pool, gt in fit_cache:
        for score, rle, feats in rescue_pool:
            label = 1 if best_iou_against_rle(rle, gt) > 0.5 else 0
            X_fit.append([score, feats["contrast"], feats["area"], feats["elongation"], feats["thickness"]])
            y_fit.append(label)
    X_fit = np.array(X_fit)
    y_fit = np.array(y_fit)
    print(f"fit pool: {len(y_fit)} candidates, {y_fit.sum()} TP ({100*y_fit.mean():.1f}%)", flush=True)
    del fit_cache

    scaler = StandardScaler().fit(X_fit)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    clf.fit(scaler.transform(X_fit), y_fit)
    print(f"classifier coefficients (score, contrast, area, elongation, thickness): "
          f"{clf.coef_[0]}", flush=True)

    print(f"\ngathering val pool from {len(val_entries)} val images...", flush=True)
    val_cache = gather_candidates(val_entries, yolo, refiner, device, per_image)

    def pq_for(accept_prob_thresh):
        pq_num = pq_den = 0.0
        tp = 0
        n_rescued = 0
        for deployed, rescue_pool, gt in val_cache:
            rescued = []
            if rescue_pool:
                X = np.array([[s, f["contrast"], f["area"], f["elongation"], f["thickness"]]
                              for s, r, f in rescue_pool])
                probs = clf.predict_proba(scaler.transform(X))[:, 1]
                rescued = [(s, r) for (s, r, f), p in zip(rescue_pool, probs) if p >= accept_prob_thresh]
            n_rescued += len(rescued)
            kept = paint_panoptic_rle(deployed + rescued)
            num, den, m_tp = pq_against_gt(kept, gt)
            pq_num += num
            pq_den += den
            tp += m_tp
        return (pq_num / pq_den if pq_den else 0.0), tp, n_rescued

    baseline_pq, baseline_tp, _ = pq_for(2.0)  # threshold above 1.0 = never rescue
    print(f"\nbaseline (no rescue): PQ={baseline_pq:.4f} TP={baseline_tp}\n")

    results = []
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pq, tp, n_rescued = pq_for(thresh)
        print(f"accept_prob>={thresh}: PQ={pq:.4f} TP={tp} (+{tp-baseline_tp} vs baseline, "
              f"avg {n_rescued/len(val_entries):.2f} rescued/image)", flush=True)
        results.append((pq, thresh, tp))

    results.sort(reverse=True)
    print("\n=== top 5 ===")
    for pq, thresh, tp in results[:5]:
        print(f"PQ={pq:.4f}  accept_prob_thresh={thresh}  TP={tp}")


if __name__ == "__main__":
    main()
