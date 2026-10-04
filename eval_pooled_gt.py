"""How would our pipeline score if the hidden test GT pools all annotators of an
image into one GT list (the submission format is keyed by image stem, not by
annotator)? Scores the val split both ways with the official Self-Evaluation
notebook's PQ logic: every (gt, pred) pair with IoU > 0.5 is a TP, a
prediction is an FP only if it hits no GT, a GT is an FN only if no
prediction hits it.

  per-entry: each (image, annotator) entry scored separately (our usual local PQ)
  pooled:    one GT list per image = all annotators' filaments together

    python eval_pooled_gt.py --cache v10_all_val.pkl --accepts 0.3 0.4 0.5 0.6
"""
import argparse
import pickle
from collections import OrderedDict

import numpy as np
import pycocotools.mask as mu

from dataset import train_val_split
from sweep_ensemble_4way import H, W, dedup_nms_rle, fit_calibrators, paint_panoptic_rle
from sweep_mask_fusion import SOURCES, fuse_image


def official_counts(pred, gt):
    """(sum IoU of hits, n_tp_pairs, n_fp, n_fn) as in the Self-Evaluation notebook."""
    if not gt:
        return 0.0, 0, len(pred), 0
    if not pred:
        return 0.0, 0, 0, len(gt)
    p = [{"size": [H, W], "counts": r.encode()} for r in pred]
    g = [{"size": [H, W], "counts": r.encode()} for r in gt]
    iou = mu.iou(p, g, [0] * len(g))  # pred x gt
    hit = iou > 0.5
    return float(iou[hit].sum()), int(hit.sum()), int((hit.sum(1) == 0).sum()), int((hit.sum(0) == 0).sum())


def pq(c):
    s, tp, fp, fn = c
    d = tp + 0.5 * fp + 0.5 * fn
    return s / d if d else 0.0


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--cache", default="v10_all_val.pkl")
    a.add_argument("--sources", nargs="+", default=SOURCES)
    a.add_argument("--accepts", type=float, nargs="+", default=[0.3, 0.4, 0.5, 0.6])
    a.add_argument("--dedup", type=float, default=0.05)
    a.add_argument("--fusion", type=float, nargs=3, default=None, help="cluster mask fusion config")
    args = a.parse_args()

    cache = pickle.load(open(args.cache, "rb"))
    _, val_entries, _ = train_val_split(val_frac=0.1, seed=0)
    assert len(val_entries) == len(cache)
    cals = fit_calibrators(cache, args.sources, list(range(len(cache))))  # in-sample, for the comparison only

    files = OrderedDict()
    for i, e in enumerate(val_entries):
        files.setdefault(e["file_name"], []).append(i)
    n_multi = sum(len(v) > 1 for v in files.values())
    print(f"{len(cache)} entries, {len(files)} images ({n_multi} with 2+ annotators)")

    for acc in args.accepts:
        per_entry, pooled = np.zeros(4), np.zeros(4)
        for fname, idxs in files.items():
            ps = cache[idxs[0]][0]  # same image -> same candidates for every annotator entry
            cands = []
            for k in args.sources:
                if ps[k]:
                    cs = cals[k].predict([s for s, _, _ in ps[k]])
                    cands += [(float(v), r) for v, (_, r, _) in zip(cs, ps[k])]
            cands.sort(key=lambda x: -x[0])
            if args.fusion:
                global_acc = acc
                import sweep_mask_fusion as smf
                smf.ACCEPT = global_acc
                cfg = tuple(args.fusion)
                kept = paint_panoptic_rle(fuse_image(cands, [cfg])[cfg])
            else:
                kept = paint_panoptic_rle(dedup_nms_rle([c for c in cands if c[0] >= acc], args.dedup))
            all_gt = []
            for i in idxs:
                gt = cache[i][1]
                per_entry += official_counts(kept, gt)
                all_gt += gt
            pooled += official_counts(kept, all_gt)
        print(f"accept={acc}: per-entry PQ={pq(per_entry):.4f} (TP {int(per_entry[1])} FP {int(per_entry[2])} "
              f"FN {int(per_entry[3])}) | pooled-GT PQ={pq(pooled):.4f} (TP {int(pooled[1])} FP {int(pooled[2])} "
              f"FN {int(pooled[3])})", flush=True)


if __name__ == "__main__":
    main()
