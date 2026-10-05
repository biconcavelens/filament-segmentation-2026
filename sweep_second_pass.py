"""Second-pass refinement of the fused leader masks: the cluster-fused mask is
more complete than any single detector proposal, so re-centring the refiner
crop on it (at the usual 1.8x context) may let the refiner follow tails the
first pass truncated. Compares, in-sample: fused masks as-is, re-refined fused
masks, and their union.

    python sweep_second_pass.py --cpu
"""
import argparse
import pickle

import numpy as np
import pycocotools.mask as mu
import torch
from PIL import Image

import sweep_grow
from dataset import train_val_split, IMG_DIR
from predict_trained import to_rle
from sweep_ensemble_4way import H, W, paint_panoptic_rle, pq_against_gt
from sweep_mask_fusion import calibrated, fuse_image
from train_refiner import load_refiner

LEADERS = ["Av", "B1280v", "Cv", "L1280v"]
VOTERS = LEADERS + ["A", "B1280", "C", "L1280", "AF", "BF", "CF", "LF"]
CFG = (0.5, 0.3, 0.3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="big_val.pkl")
    p.add_argument("--refiner", default="checkpoints/refiner_v10_resnet34_best.pt")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    if args.cpu:
        sweep_grow.DEVICE = torch.device("cpu")
    torch.set_num_threads(6)
    refiner = load_refiner(args.refiner, sweep_grow.DEVICE)
    cache = pickle.load(open(args.cache, "rb"))
    _, val_entries, _ = train_val_split(val_frac=0.1, seed=0)
    pooled_all = calibrated(cache, VOTERS, crossfit=False)
    tot = {k: np.zeros(3) for k in ["fused", "rerefined", "union", "vote2"]}
    for n, (pooled, (_, gt), e) in enumerate(zip(pooled_all, cache, val_entries), 1):
        fused = fuse_image(pooled, [CFG], LEADERS)[CFG]
        if not fused:
            for k in tot:
                tot[k] += pq_against_gt([], gt)
            continue
        gray = np.array(Image.open(IMG_DIR / e["file_name"]).convert("L"))
        out = {k: [] for k in tot}
        for s, r in fused:
            m = mu.decode({"size": [H, W], "counts": r.encode()})
            rr = sweep_grow.refine(refiner, gray, m, 1.8)
            out["fused"].append((s, r))
            out["rerefined"].append((s, to_rle(rr) if rr.sum() else r))
            out["union"].append((s, to_rle(m | rr)))
            out["vote2"].append((s, to_rle(m & rr) if (m & rr).sum() else r))  # intersection = 2-way majority
        for k in tot:
            tot[k] += pq_against_gt(paint_panoptic_rle(out[k]), gt)
        if n % 20 == 0:
            print(f"  {n}/{len(cache)}", flush=True)
    for k, t in tot.items():
        print(f"{k}: PQ={t[0] / t[1]:.4f} TP={int(t[2])}", flush=True)


if __name__ == "__main__":
    main()
