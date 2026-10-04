"""Is an out-of-fold PQ gain real, or fold luck? Re-runs the OOF comparison of
two scoring configs under several random image->fold assignments and
bootstraps the per-image PQ difference (paired, same images).

A config is (sources, scorer, accept): scorer "iso" = per-source isotonic,
"gbm" = the sweep_rescorer.py classifier. Default pairs:
  Experiment A (A+B1280+B2048, iso) vs the real-0.39 2-way (A+B1280, iso)
  GBM rescorer vs isotonic on the same 3 sources

    python rescorer_robustness.py
"""
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from sweep_ensemble_4way import CACHE_PATH, dedup_nms_rle, paint_panoptic_rle, pq_against_gt
from sweep_rescorer import image_features, GBM_PARAMS, K

DEDUP = 0.05
TWO_WAY = (("A", "B1280"), "iso", 0.45)
EXP_A = (("A", "B1280", "B2048"), "iso", 0.5)
GBM3 = (("A", "B1280", "B2048"), "gbm", 0.45)
PAIRS = [(TWO_WAY, EXP_A), (EXP_A, GBM3)]
SEEDS = [1, 2, 3, 4, 5]
N_BOOT = 2000


def oof_scores(feats, fold, scorer, n_sources):
    n = len(feats)
    out = [None] * n
    for k in range(K):
        tr = [i for i in range(n) if fold[i] != k and feats[i] is not None]
        X = np.vstack([feats[i][0] for i in tr])
        y = np.concatenate([feats[i][1] for i in tr])
        if scorer == "gbm":
            gbm = HistGradientBoostingClassifier(**GBM_PARAMS).fit(X, y)
        else:
            src = np.concatenate([feats[i][3] for i in tr])
            sc = np.concatenate([feats[i][4] for i in tr])
            isos = [IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(sc[src == j], y[src == j])
                    for j in range(n_sources)]
        for i in range(n):
            if fold[i] != k or feats[i] is None:
                continue
            if scorer == "gbm":
                out[i] = gbm.predict_proba(feats[i][0])[:, 1]
            else:
                out[i] = np.array([isos[j].predict([v])[0] for j, v in zip(feats[i][3], feats[i][4])])
    return out


def per_image_pq(scores, feats, gts, accept):
    out = []
    for s, f, gt in zip(scores, feats, gts):
        pooled = [] if f is None else [(v, r) for v, r in zip(s, f[2]) if v >= accept]
        out.append(pq_against_gt(paint_panoptic_rle(dedup_nms_rle(pooled, DEDUP)), gt)[:2])
    return np.array(out)  # (n_images, 2): PQ numerator, denominator


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=CACHE_PATH)
    p.add_argument("--base-cache", default=None, help="score the base config from a different cache")
    p.add_argument("--pairs", choices=["default", "2way"], default="default",
                   help="2way: compare the 2-way iso config across --base-cache vs --cache")
    p.add_argument("--custom", nargs=4, metavar=("BASE_SOURCES", "BASE_ACCEPT", "CAND_SOURCES", "CAND_ACCEPT"),
                   help="one iso-vs-iso pair, sources comma-separated, e.g. A,B1280 0.45 A,B1280,C 0.5; "
                        "CAND_ACCEPT may be a comma list (one pair each)")
    p.add_argument("--cand-scorer", choices=["iso", "gbm"], default="iso", help="scorer for the --custom candidate")
    args = p.parse_args()
    global PAIRS
    if args.pairs == "2way":
        PAIRS = [(TWO_WAY, TWO_WAY)]
    if args.custom:
        bs, ba, cs, ca = args.custom
        PAIRS = [((tuple(bs.split(",")), "iso", float(ba)), (tuple(cs.split(",")), args.cand_scorer, float(a)))
                 for a in ca.split(",")]
    cache = pickle.load(open(args.cache, "rb"))
    base_cache = pickle.load(open(args.base_cache, "rb")) if args.base_cache else cache
    assert all(a[1] == b[1] for a, b in zip(cache, base_cache)), "caches must share val order/GT"
    gts = [gt for _, gt in cache]
    n = len(cache)
    feats = {}  # (which cache, sources) -> per-image features
    for pair in PAIRS:
        for which, cfg in (("base", pair[0]), ("cand", pair[1])):
            src_cache = base_cache if which == "base" else cache
            if (which, cfg[0]) not in feats:
                feats[(which, cfg[0])] = [image_features(ps, list(cfg[0])) for ps, _ in src_cache]

    for base, cand in PAIRS:
        print(f"\n{'+'.join(cand[0])}/{cand[1]}@{cand[2]} [{args.cache}]  vs  "
              f"{'+'.join(base[0])}/{base[1]}@{base[2]} [{args.base_cache or args.cache}]", flush=True)
        fb, fc = feats[("base", base[0])], feats[("cand", cand[0])]
        for seed in SEEDS:
            fold = np.random.default_rng(seed).permutation(n) % K
            a = per_image_pq(oof_scores(fb, fold, base[1], len(base[0])), fb, gts, base[2])
            b = per_image_pq(oof_scores(fc, fold, cand[1], len(cand[0])), fc, gts, cand[2])
            rng = np.random.default_rng(0)
            diffs = []
            for _ in range(N_BOOT):
                idx = rng.integers(0, n, n)
                diffs.append(b[idx, 0].sum() / b[idx, 1].sum() - a[idx, 0].sum() / a[idx, 1].sum())
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            pa, pb = a[:, 0].sum() / a[:, 1].sum(), b[:, 0].sum() / b[:, 1].sum()
            print(f"  seed={seed}: base={pa:.4f} cand={pb:.4f} diff={pb - pa:+.4f} "
                  f"95% CI [{lo:+.4f}, {hi:+.4f}] P(diff>0)={np.mean(np.array(diffs) > 0):.3f}", flush=True)


if __name__ == "__main__":
    main()
