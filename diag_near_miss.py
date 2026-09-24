"""Where do Experiment A's errors come from? Replays its pipeline (in-sample
isotonic, accept 0.5, dedup 0.05) from the cached val candidates, then:
  1. buckets FN GTs / FP predictions by their best IoU (near miss vs total miss)
  2. checks instance granularity: is a near-miss GT covered by the UNION of
     several kept predictions (we fragmented it), or does one prediction
     overlap several GTs (we merged them)?
  3. PQ if every kept mask were dilated/eroded by k px (systematic width bias?)
"""
import pickle

import numpy as np
import pycocotools.mask as mu
from scipy.ndimage import binary_dilation, binary_erosion

from sweep_ensemble_4way import CACHE_PATH, H, W, fit_calibrators, dedup_nms_rle, paint_panoptic_rle, pq_against_gt
from predict_trained import to_rle

SOURCES, ACCEPT, DEDUP = ["A", "B1280", "B2048"], 0.5, 0.05


def enc(r):
    return {"size": [H, W], "counts": r.encode()}


def morph(rle, k):
    if k == 0:
        return rle
    m = mu.decode(enc(rle)).astype(bool)
    x, y, w, h = (int(v) for v in mu.toBbox(enc(rle)))
    pad = abs(k) + 1
    y0, y1, x0, x1 = max(0, y - pad), min(H, y + h + pad), max(0, x - pad), min(W, x + w + pad)
    crop = m[y0:y1, x0:x1]
    crop = binary_dilation(crop, iterations=k) if k > 0 else binary_erosion(crop, iterations=-k)
    out = np.zeros((H, W), dtype=np.uint8)
    out[y0:y1, x0:x1] = crop
    return to_rle(out)


cache = pickle.load(open(CACHE_PATH, "rb"))
cals = fit_calibrators(cache, SOURCES, list(range(len(cache))))
kept_all = []
for per_source, gt in cache:
    pooled = []
    for key in SOURCES:
        if per_source[key]:
            cs = cals[key].predict([s for s, _, _ in per_source[key]])
            pooled += [(float(c), r) for c, (_, r, _) in zip(cs, per_source[key]) if c >= ACCEPT]
    kept_all.append(paint_panoptic_rle(dedup_nms_rle(pooled, DEDUP)))

fn_b = {"near 0.3-0.5": 0, "partial 0.1-0.3": 0, "none <0.1": 0}
fp_b = {"near 0.3-0.5": 0, "partial 0.1-0.3": 0, "none <0.1": 0}
fragmented = merged = tp = n_gt = n_pred = 0
for kept, (_, gt) in zip(kept_all, cache):
    n_gt += len(gt)
    n_pred += len(kept)
    if not kept or not gt:
        fn_b["none <0.1"] += len(gt)
        fp_b["none <0.1"] += len(kept)
        continue
    P, G = [enc(r) for r in kept], [enc(r) for r in gt]
    iou = mu.iou(P, G, [0] * len(G))  # (pred, gt)
    tp += int((iou.max(axis=0) > 0.5).sum())
    for g in range(len(G)):
        b = iou[:, g].max()
        if b > 0.5:
            continue
        fn_b["near 0.3-0.5" if b >= 0.3 else "partial 0.1-0.3" if b >= 0.1 else "none <0.1"] += 1
        parts = [P[i] for i in np.where(iou[:, g] > 0.05)[0]]
        if len(parts) >= 2 and mu.iou([mu.merge(parts)], [G[g]], [0])[0, 0] > 0.5:
            fragmented += 1
    for i in range(len(P)):
        b = iou[i].max()
        if b > 0.5:
            continue
        fp_b["near 0.3-0.5" if b >= 0.3 else "partial 0.1-0.3" if b >= 0.1 else "none <0.1"] += 1
        # fraction of each GT covered by this prediction (iscrowd=1: intersection / area(first arg))
        cover = mu.iou(G, [P[i]], [1])[:, 0]
        if (cover > 0.3).sum() >= 2:
            merged += 1

print(f"GT={n_gt} kept={n_pred} TP={tp} FN={n_gt - tp} FP={n_pred - tp}")
print("FN by best IoU of any kept mask:", fn_b)
print("FP by best IoU with any GT:     ", fp_b)
print(f"FN GTs recoverable by merging >=2 of our fragments: {fragmented}")
print(f"FPs covering >=2 GTs (>30% each), i.e. merged instances: {merged}")

for k in [-2, -1, 0, 1, 2, 3]:
    num = den = 0.0
    for kept, (_, gt) in zip(kept_all, cache):
        n_, d_, _ = pq_against_gt([morph(r, k) for r in kept], gt)
        num, den = num + n_, den + d_
    print(f"morph {k:+d}px: PQ={num / den:.4f}", flush=True)
