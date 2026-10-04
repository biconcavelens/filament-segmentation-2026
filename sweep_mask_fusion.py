"""Cluster mask fusion: instead of keeping only the top-scored mask of each
group of overlapping cross-detector candidates (true-NMS dedup), fuse the
group's refined masks by a score-weighted pixel vote.

Why: the dominant remaining error is truncation -- near misses with median
precision 0.78 / recall 0.49 vs GT (diag_near_miss_shape.py) -- and the
detectors' refined masks for the same filament are truncated differently, so
a vote over the cluster can recover tails the single winner misses.

Leaders = the usual dedup survivors (calibrated score >= accept, IoU <= dedup
with earlier leaders). Every other candidate with calibrated score >=
member_floor joins the leader it overlaps most if that IoU > fuse_iou. The
fused mask keeps pixels whose score-weighted vote share >= vote_frac
(vote_frac ~0 = union of the cluster).

    python sweep_mask_fusion.py --cache v10_all_val.pkl
    python sweep_mask_fusion.py --cache v10_all_val.pkl --crossfit   # calibrators fit out-of-fold
"""
import argparse
import itertools
import pickle

import numpy as np
import pycocotools.mask as mu

from predict_trained import to_rle
from sweep_ensemble_4way import H, W, fit_calibrators, paint_panoptic_rle, pq_against_gt

SOURCES = ["A", "B1280", "C", "L1280"]
ACCEPT, DEDUP = 0.5, 0.05


def calibrated(cache, sources, crossfit, seed=0, fixed=None):
    """Per image: list of (calibrated score, rle, source). fixed: calibrators fit elsewhere (test time)."""
    n = len(cache)
    if fixed is not None:
        fold, cals = np.zeros(n, int), {0: fixed}
    elif crossfit:
        fold = np.random.default_rng(seed).permutation(n) % 2
        cals = {k: fit_calibrators(cache, sources, [i for i in range(n) if fold[i] != k]) for k in (0, 1)}
    else:
        fold, cals = np.zeros(n, int), {0: fit_calibrators(cache, sources, list(range(n)))}
    out = []
    for i, (ps, _) in enumerate(cache):
        c = cals[fold[i]]
        pooled = []
        for key in sources:
            if ps[key]:
                cs = c[key].predict([s for s, _, _ in ps[key]])
                pooled += [(float(v), r, key) for v, (_, r, _) in zip(cs, ps[key])]
        out.append(sorted(pooled, key=lambda x: -x[0]))
    return out


def fuse_image(pooled, configs, leader_sources=None):
    """Returns {config: [(score, rle)]} of fused leader masks. leader_sources: only candidates
    from these sources can lead (be detections); the rest only vote on leaders' masks."""
    if not pooled:
        return {cfg: [] for cfg in configs}
    rles = [{"size": [H, W], "counts": r.encode()} for _, r, _ in pooled]
    scores = np.array([s for s, _, _ in pooled])
    can_lead = [leader_sources is None or src in leader_sources for _, _, src in pooled]
    iou = mu.iou(rles, rles, [0] * len(rles))
    leaders = []
    for i in range(len(pooled)):
        if can_lead[i] and scores[i] >= ACCEPT and all(iou[i, j] <= DEDUP for j in leaders):
            leaders.append(i)
    decoded = {}
    boxes = mu.toBbox(rles)  # x, y, w, h

    def mask(i):
        if i not in decoded:
            decoded[i] = mu.decode(rles[i])
        return decoded[i]

    others = [j for j in range(len(pooled)) if j not in set(leaders)]
    out = {}
    for fuse_iou, floor, vote in configs:
        if not leaders:
            out[(fuse_iou, floor, vote)] = []
            continue
        members = {i: [i] for i in leaders}
        for j in others:
            if scores[j] < floor:
                continue
            ious = iou[j, leaders]
            k = int(np.argmax(ious))
            if ious[k] > fuse_iou:
                members[leaders[k]].append(j)
        fused = []
        for i in leaders:
            if len(members[i]) == 1:
                fused.append((scores[i], pooled[i][1]))
                continue
            w = scores[members[i]]
            b = boxes[members[i]]
            x0, y0 = int(b[:, 0].min()), int(b[:, 1].min())
            x1, y1 = int(np.ceil((b[:, 0] + b[:, 2]).max())), int(np.ceil((b[:, 1] + b[:, 3]).max()))
            share = sum(wi * mask(j)[y0:y1, x0:x1] for wi, j in zip(w, members[i])) / w.sum()
            m = np.zeros((H, W), dtype=np.uint8)
            m[y0:y1, x0:x1] = (share >= vote - 1e-9) & (share > 0)
            fused.append((scores[i], to_rle(m)))
        out[(fuse_iou, floor, vote)] = fused
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="v10_all_val.pkl")
    p.add_argument("--sources", nargs="+", default=SOURCES)
    p.add_argument("--crossfit", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fuse-ious", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    p.add_argument("--floors", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    p.add_argument("--votes", type=float, nargs="+", default=[0.01, 0.3, 0.5])
    p.add_argument("--robust", type=float, nargs=3, metavar=("FUSE_IOU", "FLOOR", "VOTE"),
                   help="5 crossfit seeds + paired bootstrap of this config vs plain dedup")
    p.add_argument("--predict", nargs=4, metavar=("TEST_CACHE", "FUSE_IOU", "FLOOR", "VOTE"),
                   help="write a submission: calibrate on --cache, fuse the test cache with this config")
    p.add_argument("--out", default="submission_maskfusion.csv")
    p.add_argument("--leaders", nargs="+", default=None, help="sources allowed to lead; others only vote")
    args = p.parse_args()

    cache = pickle.load(open(args.cache, "rb"))
    if args.predict:
        import pandas as pd
        test = pickle.load(open(args.predict[0], "rb"))
        cfg = tuple(float(v) for v in args.predict[1:])
        cals = fit_calibrators(cache, args.sources, list(range(len(cache))))
        rows = []
        for pooled, (_, stem) in zip(calibrated(test, args.sources, False, fixed=cals), test):
            kept = paint_panoptic_rle(fuse_image(pooled, [cfg], args.leaders)[cfg])
            rows.extend({"filament_id": f"{stem}_{k}", "segmentation_rle": r} for k, r in enumerate(kept, 1))
        pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(args.out, index=False)
        print(f"wrote {args.out}: {len(rows)} rows, {len(test)} images", flush=True)
        return
    if args.robust:
        cfg = tuple(args.robust)
        for seed in [1, 2, 3, 4, 5]:
            pooled_all = calibrated(cache, args.sources, True, seed)
            a, b = [], []
            for pooled, (_, gt) in zip(pooled_all, cache):
                res = fuse_image(pooled, [cfg, (2.0, 2.0, 1.0)], args.leaders)
                b.append(pq_against_gt(paint_panoptic_rle(res[cfg]), gt)[:2])
                a.append(pq_against_gt(paint_panoptic_rle(res[(2.0, 2.0, 1.0)]), gt)[:2])
            a, b = np.array(a), np.array(b)
            rng = np.random.default_rng(0)
            diffs = []
            for _ in range(2000):
                idx = rng.integers(0, len(a), len(a))
                diffs.append(b[idx, 0].sum() / b[idx, 1].sum() - a[idx, 0].sum() / a[idx, 1].sum())
            pa, pb = a[:, 0].sum() / a[:, 1].sum(), b[:, 0].sum() / b[:, 1].sum()
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            print(f"seed={seed}: base={pa:.4f} fused={pb:.4f} diff={pb - pa:+.4f} 95% CI [{lo:+.4f}, {hi:+.4f}] "
                  f"P(diff>0)={np.mean(np.array(diffs) > 0):.3f}", flush=True)
        return
    pooled_all = calibrated(cache, args.sources, args.crossfit, args.seed)
    configs = list(itertools.product(args.fuse_ious, args.floors, args.votes))
    base = np.zeros(3)
    tot = {cfg: np.zeros(3) for cfg in configs}
    for n, (pooled, (_, gt)) in enumerate(zip(pooled_all, cache), 1):
        res = fuse_image(pooled, configs, args.leaders)
        leaders_only = fuse_image(pooled, [(2.0, 2.0, 1.0)], args.leaders)[(2.0, 2.0, 1.0)]  # no members = plain dedup
        base += pq_against_gt(paint_panoptic_rle(leaders_only), gt)
        for cfg in configs:
            tot[cfg] += pq_against_gt(paint_panoptic_rle(res[cfg]), gt)
        if n % 20 == 0:
            print(f"  {n}/{len(cache)}", flush=True)
    print(f"baseline (dedup only): PQ={base[0] / base[1]:.4f} TP={int(base[2])}", flush=True)
    for cfg, t in sorted(tot.items(), key=lambda kv: -kv[1][0] / kv[1][1]):
        print(f"fuse_iou={cfg[0]} floor={cfg[1]} vote={cfg[2]}: PQ={t[0] / t[1]:.4f} TP={int(t[2])}", flush=True)


if __name__ == "__main__":
    main()
