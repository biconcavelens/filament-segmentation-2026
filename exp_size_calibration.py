"""Size-aware calibration: one isotonic map per (source, candidate-area band) instead of per source.
Motivation: at the same raw score, small candidates are TPs less often than large ones for the
YOLOs, RT-DETR and the semantic model (not for Mask R-CNN). Band edges are area quantiles of the
fitting half's candidates, so nothing is fitted on the scored half. Compared with the per-source
calibration on the paper's five out-of-fold splits, inside the full system (paired bootstrap).

    python exp_size_calibration.py [n_bands]
"""
import pickle
import sys

import numpy as np
import pycocotools.mask as mu
from sklearn.isotonic import IsotonicRegression

from eval_pooled_gt import official_counts
from paper_oof import CFG, DET, SEEDS, VOTE12, pq_of
from sweep_ensemble_4way import H, W, paint_panoptic_rle
from sweep_mask_fusion import calibrated, fuse_image

SOURCES, LEADERS = VOTE12 + ["S"], DET + ["S"]


def calibrated_by_size(cache, areas, sources, seed, n_bands):
    """Like sweep_mask_fusion.calibrated(crossfit=True) but with per-(source, area band) isotonic maps."""
    n = len(cache)
    fold = np.random.default_rng(seed).permutation(n) % 2  # same folds as calibrated()
    out = [[] for _ in range(n)]
    for k in (0, 1):
        fit = [i for i in range(n) if fold[i] != k]
        for src in sources:
            a = np.concatenate([areas[i][src] for i in fit])
            s = np.array([c[0] for i in fit for c in cache[i][0][src]])
            y = np.array([c[2] for i in fit for c in cache[i][0][src]])
            edges = np.quantile(a, np.linspace(0, 1, n_bands + 1)[1:-1])
            band = np.searchsorted(edges, a)
            cals = [IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(s[band == b], y[band == b])
                    for b in range(n_bands)]
            for i in (i for i in range(n) if fold[i] == k):
                cands = cache[i][0][src]
                if not cands:
                    continue
                bi = np.searchsorted(edges, areas[i][src])
                out[i] += [(float(cals[b].predict([c[0]])[0]), c[1], src) for b, c in zip(bi, cands)]
    return [sorted(p, key=lambda x: -x[0]) for p in out]


def main():
    n_bands = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    cache = pickle.load(open("paper_val.pkl", "rb"))
    areas = [{src: mu.area([{"size": [H, W], "counts": r.encode()} for _, r, _ in ps[src]]).astype(float)
              if ps[src] else np.zeros(0) for src in SOURCES} for ps, _ in cache]
    boot = np.random.default_rng(0).integers(0, len(cache), (2000, len(cache)))
    base, new = np.zeros((len(SEEDS), len(cache), 4)), np.zeros((len(SEEDS), len(cache), 4))
    for k, seed in enumerate(SEEDS):
        for tot, pooled_all in [(base, calibrated(cache, SOURCES, True, seed)),
                                (new, calibrated_by_size(cache, areas, SOURCES, seed, n_bands))]:
            for i, (pooled, (_, gt)) in enumerate(zip(pooled_all, cache)):
                tot[k, i] = official_counts(paint_panoptic_rle(fuse_image(pooled, [CFG], LEADERS)[CFG]), gt)
        d = pq_of(new[k][boot].sum(1)) - pq_of(base[k][boot].sum(1))
        print(f"seed {seed}: per-source {pq_of(base[k].sum(0)):.4f}  by size ({n_bands} bands) "
              f"{pq_of(new[k].sum(0)):.4f}  diff {pq_of(new[k].sum(0)) - pq_of(base[k].sum(0)):+.4f} "
              f"[{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]  "
              f"TP {base[k].sum(0)[1]:.0f}->{new[k].sum(0)[1]:.0f} FP {base[k].sum(0)[2]:.0f}->{new[k].sum(0)[2]:.0f}",
              flush=True)


if __name__ == "__main__":
    main()
