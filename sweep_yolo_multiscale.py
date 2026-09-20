"""Multi-scale TTA for YOLO: pool candidates from imgsz=1280 (the validated
scale) and imgsz=2048 (native resolution, no downsampling) using the SAME
true-NMS-dedup mechanism validated for cross-detector fusion this session
(dedup_nms_rle in sweep_ensemble_true_dedup.py / sweep_ensemble_3way.py).

Motivation: established this session that detector RECALL (not refiner mask
quality -- spine-weight sweep was flat) is the dominant remaining PQ loss.
Downsampling 2048x2048 -> 1280 for detection loses signal disproportionately
for thin/faint filaments (SAHI/tiled-inference literature: resizing
significantly hurts small/thin object detection). A blind imgsz bump was
already tried and regressed on the real leaderboard (RESULTS.md: imgsz 1536/
1792 looked better locally, regressed real score) -- but that was tested as
a single-scale REPLACEMENT via a 2D imgsz x conf grid (established
overfit-prone pattern). This instead MERGES both scales as complementary
candidate sources via the same score-sort-greedy-discard dedup already
proven to transfer (cross-detector ensemble: 0.4252 -> 0.4407 real 0.38 ->
0.39), rather than blindly replacing one scale with another.
"""
import pickle
from pathlib import Path

import numpy as np
import torch
import pycocotools.mask as mu
from PIL import Image
from ultralytics import YOLO
from sklearn.isotonic import IsotonicRegression

from dataset import train_val_split, IMG_DIR, H, W
from train_refiner import RefinerUNet
from crop_dataset import square_bounds, CROP_SIZE
from predict_trained import to_rle
from predict_refined import refine_with_tta

YOLO_CKPT = "kaggle_dataset_upload/yolo11m_cls_best.pt"
REFINER_CKPT = "checkpoints/refiner_v5_best.pt"
FLOOR = 0.05
MIN_AREA = 20
SCALES = [1280, 2048]


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


def _candidates_at_scale(model, img_path, imgsz, floor):
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
        if int(remaining.sum()) < min_area:
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


CACHE_PATH = Path("scratch_multiscale_cache.pkl")


def main():
    if CACHE_PATH.exists():
        print(f"loading cached merged candidates from {CACHE_PATH}", flush=True)
        with open(CACHE_PATH, "rb") as f:
            merged_cache = pickle.load(f)
        run_sweep(merged_cache)
        return

    device = torch.device("cuda")
    yolo = YOLO(YOLO_CKPT)

    rstate = torch.load(REFINER_CKPT, map_location=device)
    refiner = RefinerUNet(in_channels=rstate.get("in_channels", 1),
                           out_channels=rstate.get("out_channels", 1)).to(device)
    refiner.load_state_dict(rstate["model"])
    refiner.eval()

    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)

    per_image_cache = {s: [] for s in SCALES}
    scores_by_scale = {s: [] for s in SCALES}
    labels_by_scale = {s: [] for s in SCALES}

    with torch.no_grad():
        for i, e in enumerate(val_entries, 1):
            img_path = IMG_DIR / e["file_name"]
            gray = np.array(Image.open(img_path).convert("L"))

            gt = []
            for a in per_image.get(e["id"], []):
                rles = mu.frPyObjects(a["segmentation"], H, W)
                gt.append(to_rle(mu.decode(mu.merge(rles))))
            gt_dicts = [{"size": [H, W], "counts": r.encode()} for r in gt]

            for scale in SCALES:
                cands = _candidates_at_scale(yolo, img_path, scale, FLOOR)
                refined = []
                for score, coarse in cands:
                    ref = refine_candidate(refiner, device, gray, coarse)
                    if ref.sum() == 0:
                        continue
                    label = 0
                    if gt_dicts:
                        pred_rle = {"size": [H, W], "counts": to_rle(ref).encode()}
                        iou = mu.iou([pred_rle], gt_dicts, [0] * len(gt_dicts))
                        label = 1 if float(iou.max()) > 0.5 else 0
                    refined.append((score, to_rle(ref)))
                    scores_by_scale[scale].append(score)
                    labels_by_scale[scale].append(label)
                per_image_cache[scale].append((refined, gt))

            if i % 20 == 0:
                print(f"  cached {i}/{len(val_entries)}", flush=True)

    cals = {}
    for scale in SCALES:
        n_tp = sum(labels_by_scale[scale])
        print(f"scale={scale}: {len(scores_by_scale[scale])} candidates ({n_tp} TP)")
        cal = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        cal.fit(scores_by_scale[scale], labels_by_scale[scale])
        cals[scale] = cal

    n = len(val_entries)
    merged_cache = []
    for idx in range(n):
        pooled = []
        gt = per_image_cache[SCALES[0]][idx][1]
        for scale in SCALES:
            refined, _ = per_image_cache[scale][idx]
            if refined:
                cs = cals[scale].predict([s for s, _ in refined])
                pooled.extend((float(c), r) for c, (_, r) in zip(cs, refined))
        merged_cache.append((pooled, gt))

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(merged_cache, f)
    print(f"cached merged candidates to {CACHE_PATH}", flush=True)

    run_sweep(merged_cache)


def run_sweep(merged_cache):
    def pq_for(dedup_iou, accept_thresh):
        pq_num = pq_den = 0.0
        tp = 0
        for pooled, gt in merged_cache:
            filtered = [(s, r) for s, r in pooled if s >= accept_thresh]
            deduped = dedup_nms_rle(filtered, dedup_iou)
            kept = paint_panoptic_rle(deduped)
            num, den, m_tp = pq_against_gt(kept, gt)
            pq_num += num
            pq_den += den
            tp += m_tp
        return (pq_num / pq_den if pq_den else 0.0), tp

    print("\n=== baseline: YOLO solo imgsz=1280 PQ=0.4252 ===\n")
    results = []
    for dedup_iou in [0.03, 0.05, 0.08, 0.1, 0.15]:
        for accept_thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
            pq, tp = pq_for(dedup_iou, accept_thresh)
            print(f"dedup_iou={dedup_iou} accept={accept_thresh}: PQ={pq:.4f} TP={tp}", flush=True)
            results.append((pq, dedup_iou, accept_thresh, tp))

    results.sort(reverse=True)
    print("\n=== top 5 ===")
    for pq, dedup_iou, accept_thresh, tp in results[:5]:
        print(f"PQ={pq:.4f}  dedup_iou={dedup_iou}  accept={accept_thresh}  TP={tp}")


if __name__ == "__main__":
    main()
