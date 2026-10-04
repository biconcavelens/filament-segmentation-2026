"""Where do the deployed v10 4-way's false negatives come from? For every GT
filament: does the candidate pool contain a mask with IoU > 0.5 (oracle hit),
and if the pipeline still misses it, was the best such candidate below the
accept threshold, or suppressed by dedup in favour of a different
(wrong) higher-scored overlapping candidate?

    python diag_selection.py --cache v10_all_val.pkl
"""
import argparse
import pickle
from collections import Counter

import numpy as np
import pycocotools.mask as mu

from sweep_ensemble_4way import H, W, dedup_nms_rle, paint_panoptic_rle
from sweep_mask_fusion import SOURCES, ACCEPT, DEDUP, calibrated


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="v10_all_val.pkl")
    args = p.parse_args()
    cache = pickle.load(open(args.cache, "rb"))
    pooled_all = calibrated(cache, SOURCES, crossfit=True)
    why = Counter()
    best_scores, leader_ious = [], []
    for pooled, (_, gt) in zip(pooled_all, cache):
        if not gt:
            continue
        gtd = [{"size": [H, W], "counts": r.encode()} for r in gt]
        kept = paint_panoptic_rle(dedup_nms_rle([(sc, r) for sc, r, _ in pooled if sc >= ACCEPT], DEDUP))
        hit = np.zeros(len(gt), bool)
        if kept:
            iou_k = mu.iou([{"size": [H, W], "counts": r.encode()} for r in kept], gtd, [0] * len(gt))
            hit = iou_k.max(axis=0) > 0.5
        if not pooled:
            why["no candidates"] += int((~hit).sum())
            continue
        rles = [{"size": [H, W], "counts": r.encode()} for _, r, _ in pooled]
        sc = np.array([s for s, _, _ in pooled])
        iou_c = mu.iou(rles, gtd, [0] * len(gt))  # candidates x gt
        leaders = [i for i, (s, _, _) in enumerate(pooled) if s >= ACCEPT]
        for g in np.where(~hit)[0]:
            good = np.where(iou_c[:, g] > 0.5)[0]
            if not good.size:
                why["no oracle candidate (detection miss / near-miss masks only)"] += 1
                leader_ious.append(iou_c[:, g].max())
                continue
            b = good[np.argmax(sc[good])]
            best_scores.append(sc[b])
            if sc[b] < ACCEPT:
                why["oracle candidate below accept"] += 1
            else:
                why["oracle candidate accepted but suppressed/repainted"] += 1
    n_fn = sum(why.values())
    print(f"FN total {n_fn}")
    for k, v in why.most_common():
        print(f"  {k}: {v}")
    bs = np.array(best_scores)
    if bs.size:
        print("best oracle candidate calibrated score, quantiles 10/25/50/75/90:",
              np.round(np.quantile(bs, [.1, .25, .5, .75, .9]), 3))
    li = np.array(leader_ious)
    if li.size:
        print("no-oracle FNs: best candidate IoU quantiles 10/25/50/75/90:",
              np.round(np.quantile(li, [.1, .25, .5, .75, .9]), 3), f"(>0.3: {(li > 0.3).sum()})")


if __name__ == "__main__":
    main()
