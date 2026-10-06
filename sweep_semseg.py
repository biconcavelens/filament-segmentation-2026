"""Instances from the semantic model's probability maps (kaggle_kernel_semseg:
fg + spine heads), scored standalone on the val split, and exported as a
candidate cache source "S" (score = mean fg probability inside the instance)
so the ensemble machinery can use it as a detection source or a voter.

Instance extraction: foreground = fg > t_fg; seeds = connected components of
(spine > t_sp) inside the foreground, bridged by a small closing; watershed on
-fg floods the foreground from the seeds; foreground components without a seed
become their own instance; instances smaller than min_area are dropped.

    python sweep_semseg.py --probs kaggle_out_semseg/probs            # grid, standalone val PQ
    python sweep_semseg.py --probs ... --export 0.5 0.5 3 60          # write semseg_val/test caches
"""
import argparse
import itertools
import pickle
from pathlib import Path

import numpy as np
import pycocotools.mask as mu
from scipy import ndimage
from skimage.segmentation import watershed

from dataset import train_val_split
from predict_trained import to_rle
from sweep_ensemble_4way import H, W, TEST_DIR, pq_against_gt

VAL_CACHE = "v10_all_val.pkl"  # only for the val entry order / GT


def load_probs(probs_dir, stem):
    """probs_dir may be a comma list of model output dirs: their maps are averaged."""
    fg = sp = 0.0
    dirs = str(probs_dir).split(",")
    for d in dirs:
        z = np.load(Path(d) / f"{stem}.npz")
        fg = fg + z["fg"].astype(np.float32) / 255.0
        sp = sp + z["spine"].astype(np.float32) / 255.0
    return fg / len(dirs), sp / len(dirs)


def instances(fg, sp, t_fg, t_sp, close_px, min_area, use_spine=True):
    """-> list of (score, binary mask) sorted by score desc."""
    fgb = fg > t_fg
    if not fgb.any():
        return []
    if use_spine:
        seeds = (sp > t_sp) & fgb
        if close_px:
            seeds = ndimage.binary_closing(seeds, iterations=close_px) & fgb
        markers, n = ndimage.label(seeds)
        lab = watershed(-fg, markers=markers, mask=fgb) if n else np.zeros(fg.shape, np.int32)
        rest, m = ndimage.label(fgb & (lab == 0))  # unseeded foreground blobs
        lab = np.where(rest > 0, rest + lab.max(), lab)
    else:
        if close_px:
            fgb = ndimage.binary_closing(fgb, iterations=close_px)
        lab, _ = ndimage.label(fgb)
    out = []
    objs = ndimage.find_objects(lab)
    for k, sl in enumerate(objs, 1):
        if sl is None:
            continue
        m = lab[sl] == k
        if m.sum() < min_area:
            continue
        full = np.zeros((H, W), np.uint8)
        full[sl][m] = 1
        out.append((float(fg[sl][m].mean()), full))
    return sorted(out, key=lambda x: -x[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probs", default="kaggle_out_semseg/probs")
    p.add_argument("--t-fg", type=float, nargs="+", default=[0.4, 0.5, 0.6])
    p.add_argument("--t-sp", type=float, nargs="+", default=[0.3, 0.5])
    p.add_argument("--close", type=int, nargs="+", default=[0, 3])
    p.add_argument("--min-area", type=int, nargs="+", default=[30, 100])
    p.add_argument("--score-floor", type=float, nargs="+", default=[0.0, 0.6, 0.7])
    p.add_argument("--export", type=float, nargs=4, metavar=("T_FG", "T_SP", "CLOSE", "MIN_AREA"))
    p.add_argument("--key", default="S", help="source key / file tag for --export")
    args = p.parse_args()
    val_cache = pickle.load(open(VAL_CACHE, "rb"))
    _, val_entries, _ = train_val_split(val_frac=0.1, seed=0)
    stems = [Path(e["file_name"]).stem for e in val_entries]

    if args.export:
        t_fg, t_sp, close, min_area = args.export
        cache = {}
        for split, items in [("val", list(zip(stems, [gt for _, gt in val_cache]))),
                             ("test", [(pth.stem, None) for pth in sorted(TEST_DIR.iterdir())])]:
            out, memo = [], {}
            for stem, gt in items:
                if stem not in memo:
                    fg, sp = load_probs(args.probs, stem)
                    memo = {stem: [(s, to_rle(m)) for s, m in instances(fg, sp, t_fg, t_sp, int(close), int(min_area))]}
                cands = memo[stem]
                if gt is not None:
                    gtd = [{"size": [H, W], "counts": r.encode()} for r in gt]
                    lab = (mu.iou([{"size": [H, W], "counts": r.encode()} for _, r in cands], gtd, [0] * len(gtd))
                           .max(axis=1) > 0.5) if cands and gtd else np.zeros(len(cands), bool)
                    out.append(({args.key: [(s, r, int(l)) for (s, r), l in zip(cands, lab)]}, gt))
                else:
                    out.append(({args.key: [(s, r, 0) for s, r in cands]}, stem))
            name = f"semseg_{split}.pkl" if args.key == "S" else f"semseg_{args.key}_{split}.pkl"
            pickle.dump(out, open(name, "wb"))
            print(f"wrote {name} ({len(out)} entries)", flush=True)
        return

    configs = list(itertools.product(args.t_fg, args.t_sp, args.close, args.min_area))
    tot = {(c, f): np.zeros(3) for c in configs for f in args.score_floor}
    memo_stem, memo = None, {}
    for n, (stem, (_, gt)) in enumerate(zip(stems, val_cache), 1):
        if stem != memo_stem:  # entries of the same image are adjacent? not guaranteed -> recompute per new stem
            fg, sp = load_probs(args.probs, stem)
            memo = {c: [(s, to_rle(m)) for s, m in instances(fg, sp, *c[:2], c[2], c[3])] for c in configs}
            memo_stem = stem
        for c in configs:
            for f in args.score_floor:
                tot[(c, f)] += pq_against_gt([r for s, r in memo[c] if s >= f], gt)
        if n % 20 == 0:
            print(f"  {n}/{len(stems)}", flush=True)
    for (c, f), t in sorted(tot.items(), key=lambda kv: -kv[1][0] / kv[1][1])[:25]:
        print(f"t_fg={c[0]} t_sp={c[1]} close={c[2]} min_area={c[3]} floor={f}: PQ={t[0] / t[1]:.4f} TP={int(t[2])}",
              flush=True)


if __name__ == "__main__":
    main()
