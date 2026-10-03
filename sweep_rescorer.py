"""Second-stage learned rescorer over the pooled ensemble candidates.

Oracle-headroom check on the cached val candidates: at floor 0.05 the
A+B1280+B2048 pool already matches 798/908 GT filaments (87.9%) at IoU>0.5
(825 with RT-DETR added), but the per-source-isotonic pipeline keeps only
590. The gap is selection, not proposals: every candidate is judged by its
own detector's score alone. This scores each candidate with cross-detector
agreement, shape, disk-position and within-image rank features via a small
gradient-boosted classifier predicting P(IoU>0.5 with some GT).

Evaluated out-of-fold by image (K folds): no image is scored by a model
trained on it. The per-source isotonic baseline is re-run on the SAME folds
so the comparison is like-for-like.

    python sweep_rescorer.py --sources A B1280 B2048
"""
import argparse
import pickle

import numpy as np
import pycocotools.mask as mu
from PIL import Image
from scipy.ndimage import binary_dilation
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

from dataset import train_val_split, IMG_DIR
from sweep_ensemble_4way import CACHE_PATH, ALL_SOURCES, H, W, dedup_nms_rle, paint_panoptic_rle, pq_against_gt

K = 4
RING = 15  # px band around the mask used as its local background
GBM_PARAMS = dict(max_iter=300, learning_rate=0.05, max_depth=4, min_samples_leaf=40,
                  l2_regularization=1.0, random_state=0)
ACCEPTS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
DEDUPS = [0.05, 0.1]


def contrast_features(rle, gray, disk_median):
    """Filaments are dark: mean intensity inside the mask vs a RING-px band around it."""
    x, y, w, h = (int(v) for v in mu.toBbox(rle))
    x0, y0 = max(0, x - RING), max(0, y - RING)
    x1, y1 = min(W, x + w + RING + 1), min(H, y + h + RING + 1)
    m = mu.decode(rle)[y0:y1, x0:x1].astype(bool)
    crop = gray[y0:y1, x0:x1].astype(np.float32)
    ring = binary_dilation(m, iterations=RING) & ~m
    if not m.any() or not ring.any():
        return [0.0] * 5
    inside, outside = crop[m].mean(), crop[ring].mean()
    return [inside, outside, outside - inside, (outside - inside) / disk_median, crop[m].std()]


def image_features(per_source, sources, gray=None):
    """One feature row per candidate: source one-hot, raw score, within-source
    rank/count, mask shape, radial position, overlap counts, for every source
    the best-overlapping candidate's IoU and raw score, and (if the image is
    given) inside-vs-ring contrast."""
    cands = [(s, score, rle, label) for s in sources for score, rle, label in per_source[s]]
    if not cands:
        return None
    rles = [{"size": [H, W], "counts": r.encode()} for _, _, r, _ in cands]
    iou = mu.iou(rles, rles, [0] * len(rles))
    np.fill_diagonal(iou, 0.0)
    areas = np.concatenate([mu.area(rles[i:i + 200]) for i in range(0, len(rles), 200)]).astype(float)  # pycocotools overflows past 255
    x, y, w, h = mu.toBbox(rles).T
    src = np.array([sources.index(s) for s, *_ in cands])
    scores = np.array([sc for _, sc, _, _ in cands])

    disk_median = max(float(np.median(gray[gray > 20])), 1.0) if gray is not None else None
    rows = []
    for i in range(len(cands)):
        same = src == src[i]
        f = [float(src[i] == k) for k in range(len(sources))]
        f += [scores[i], (scores[same] > scores[i]).sum() / same.sum(), same.sum(),
              np.log1p(areas[i]), w[i], h[i], areas[i] / max(w[i] * h[i], 1.0),
              max(w[i], h[i]) / max(min(w[i], h[i]), 1.0),
              np.hypot(x[i] + w[i] / 2 - W / 2, y[i] + h[i] / 2 - H / 2) / (W / 2),
              (iou[i] > 0.3).sum(), (iou[i] > 0.5).sum()]
        for k in range(len(sources)):
            m = iou[i][src == k]
            if m.size:
                j = int(np.argmax(m))
                f += [m[j], scores[src == k][j]]
            else:
                f += [0.0, 0.0]
        if gray is not None:
            f += contrast_features(rles[i], gray, disk_median)
        rows.append(f)
    labels = np.array([l for *_, l in cands])
    return np.array(rows), labels, [r for _, _, r, _ in cands], src, scores


def max_iou_with_gt(rles, gt):
    if not gt:
        return np.zeros(len(rles))
    pred = [{"size": [H, W], "counts": r.encode()} for r in rles]
    gt_d = [{"size": [H, W], "counts": r.encode()} for r in gt]
    return mu.iou(pred, gt_d, [0] * len(gt_d)).max(axis=1)


def sweep(name, pooled_by_image, gts):
    best = []
    for d in DEDUPS:
        for a in ACCEPTS:
            num = den = 0.0
            tp = 0
            for pooled, gt in zip(pooled_by_image, gts):
                kept = paint_panoptic_rle(dedup_nms_rle([(s, r) for s, r in pooled if s >= a], d))
                n_, d_, t_ = pq_against_gt(kept, gt)
                num, den, tp = num + n_, den + d_, tp + t_
            best.append((num / den if den else 0.0, d, a, tp))
            print(f"{name} dedup={d} accept={a}: PQ={best[-1][0]:.4f} TP={tp}", flush=True)
    best.sort(reverse=True)
    print(f"=== {name} best: PQ={best[0][0]:.4f} dedup={best[0][1]} accept={best[0][2]} TP={best[0][3]}\n",
          flush=True)
    return best[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="+", default=["A", "B1280", "B2048"], choices=ALL_SOURCES)
    p.add_argument("--contrast", action="store_true")
    p.add_argument("--target", choices=["tp", "iou"], default="tp",
                   help="tp: classify IoU>0.5; iou: regress the candidate's best IoU with GT")
    p.add_argument("--skip-isotonic", action="store_true")
    args = p.parse_args()
    sources = args.sources

    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    if args.contrast:
        _, val_entries, _ = train_val_split(val_frac=0.1, seed=0)  # same order the cache was built in
        grays = [np.array(Image.open(IMG_DIR / e["file_name"]).convert("L")) for e in val_entries]
    else:
        grays = [None] * len(cache)
    feats = [image_features(per_source, sources, g) for (per_source, _), g in zip(cache, grays)]
    print(f"contrast={args.contrast}", flush=True)
    gts = [gt for _, gt in cache]
    fold = np.arange(len(cache)) % K
    ious = [max_iou_with_gt(f[2], gt) if f is not None else None for f, gt in zip(feats, gts)]

    oof_gbm = [None] * len(cache)
    oof_iso = [None] * len(cache)
    for k in range(K):
        train = [i for i in range(len(cache)) if fold[i] != k and feats[i] is not None]
        test = [i for i in range(len(cache)) if fold[i] == k and feats[i] is not None]
        X = np.vstack([feats[i][0] for i in train])
        y = np.concatenate([feats[i][1] for i in train])
        if args.target == "iou":
            gbm = HistGradientBoostingRegressor(**GBM_PARAMS)
            gbm.fit(X, np.concatenate([ious[i] for i in train]))
        else:
            gbm = HistGradientBoostingClassifier(**GBM_PARAMS)
            gbm.fit(X, y)
        isos = {}
        for s_idx in range(len(sources)):
            src_tr = np.concatenate([feats[i][3] for i in train])
            sc_tr = np.concatenate([feats[i][4] for i in train])
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(sc_tr[src_tr == s_idx], y[src_tr == s_idx])
            isos[s_idx] = iso
        for i in test:
            Xi, _, rles, src, sc = feats[i]
            oof_gbm[i] = gbm.predict(Xi) if args.target == "iou" else gbm.predict_proba(Xi)[:, 1]
            oof_iso[i] = np.array([isos[s].predict([v])[0] for s, v in zip(src, sc)])

    idx = [i for i in range(len(cache)) if feats[i] is not None]
    y_all = np.concatenate([feats[i][1] for i in idx])
    print(f"sources={sources} candidates={len(y_all)} positives={int(y_all.sum())}")
    print(f"OOF average precision: isotonic={average_precision_score(y_all, np.concatenate([oof_iso[i] for i in idx])):.4f} "
          f"gbm={average_precision_score(y_all, np.concatenate([oof_gbm[i] for i in idx])):.4f}\n", flush=True)

    def pooled(oof):
        return [list(zip(oof[i], feats[i][2])) if feats[i] is not None else [] for i in range(len(cache))]

    if not args.skip_isotonic:
        sweep("isotonic", pooled(oof_iso), gts)
    sweep(f"gbm-{args.target}", pooled(oof_gbm), gts)


if __name__ == "__main__":
    main()
