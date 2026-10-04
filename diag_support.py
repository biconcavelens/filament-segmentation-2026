"""Do multi-source-supported low-score detections deserve acceptance? Extends
the leader set down to a low accept, and for each leader reports its TP rate
(per annotator entry, as the PQ counts it) by calibrated-score band and by
"support": the number of distinct *other* sources with a candidate at IoU > 0.5
to it. A leader is worth accepting when its TP rate exceeds roughly
0.5 * PQ / mean-TP-IoU (~0.34 here).

    python diag_support.py --cache big_val.pkl
"""
import argparse
import pickle
from collections import defaultdict

import numpy as np
import pycocotools.mask as mu

from sweep_ensemble_4way import H, W
from sweep_mask_fusion import calibrated

LEADERS = ["Av", "B1280v", "Cv", "L1280v"]
VOTERS = LEADERS + ["A", "B1280", "C", "L1280", "AF", "BF", "CF", "LF"]
DEDUP = 0.05


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="big_val.pkl")
    p.add_argument("--low", type=float, default=0.2)
    args = p.parse_args()
    cache = pickle.load(open(args.cache, "rb"))
    pooled_all = calibrated(cache, VOTERS, crossfit=True)
    stats = defaultdict(lambda: [0, 0])  # (band, support) -> [n, tp]
    for pooled, (_, gt) in zip(pooled_all, cache):
        if not pooled:
            continue
        rles = [{"size": [H, W], "counts": r.encode()} for _, r, _ in pooled]
        sc = np.array([s for s, _, _ in pooled])
        src = [s for _, _, s in pooled]
        iou = mu.iou(rles, rles, [0] * len(rles))
        leaders = []
        for i in range(len(pooled)):
            if src[i] in LEADERS and sc[i] >= args.low and all(iou[i, j] <= DEDUP for j in leaders):
                leaders.append(i)
        if gt and leaders:
            gtd = [{"size": [H, W], "counts": r.encode()} for r in gt]
            tp = mu.iou([rles[i] for i in leaders], gtd, [0] * len(gt)).max(axis=1) > 0.5
        else:
            tp = np.zeros(len(leaders), bool)
        for k, i in enumerate(leaders):
            support = len({src[j] for j in range(len(pooled))
                           if j != i and src[j] != src[i] and iou[i, j] > 0.5 and sc[j] >= 0.1})
            band = "0.2-0.3" if sc[i] < 0.3 else "0.3-0.4" if sc[i] < 0.4 else "0.4-0.5" if sc[i] < 0.5 else ">=0.5"
            sup = "0-2" if support <= 2 else "3-5" if support <= 5 else "6-8" if support <= 8 else "9+"
            stats[(band, sup)][0] += 1
            stats[(band, sup)][1] += int(tp[k])
    for band in ["0.2-0.3", "0.3-0.4", "0.4-0.5", ">=0.5"]:
        row = []
        for sup in ["0-2", "3-5", "6-8", "9+"]:
            n, t = stats[(band, sup)]
            row.append(f"support {sup}: {t}/{n} = {t / n:.2f}" if n else f"support {sup}: -")
        print(f"score {band}: " + " | ".join(row))


if __name__ == "__main__":
    main()
