"""Semantic + instance fusion: complete truncated instances from a full-image
semantic U-Net.

diag_near_miss*.py: the dominant error is ~90 near misses whose masks trace
the dark core and stop where the annotated filament continues into a fainter
tail. Fixing it inside the refiner failed (v7/v8). The semantic U-Net
(train_unet.py) predicts filament/not-filament per pixel over the whole image
-- it cannot split touching instances (0.32 real on its own) but it doesn't
need to here: each kept instance is watershed-grown into semantic foreground
connected to it (within D px), pixels going to the nearest instance, and
optionally semantic blobs touching no instance become new instances.

    python sweep_semantic_fusion.py --unet checkpoints/unet_best.pt           # val prob maps (GPU) + sweep
    python sweep_semantic_fusion.py --probs unet_val_probs.npy                # sweep only (CPU)
"""
import argparse
import pickle

import numpy as np
import pycocotools.mask as mu
import torch
from PIL import Image
from scipy import ndimage
from skimage.segmentation import watershed

from dataset import train_val_split, IMG_DIR
from predict_trained import to_rle
from sweep_ensemble_4way import H, W, fit_calibrators, dedup_nms_rle, paint_panoptic_rle, pq_against_gt

PROB_SIZE = 1024  # stored at the U-Net's input resolution, upsampled on use


def val_prob_maps(unet_ckpt, out_path):
    from predict_unet import load_model, predict_prob_map
    device = torch.device("cuda")
    model = load_model(unet_ckpt, device)
    _, val_entries, _ = train_val_split(val_frac=0.1, seed=0)
    probs = np.zeros((len(val_entries), PROB_SIZE, PROB_SIZE), np.uint8)
    for i, e in enumerate(val_entries):
        gray = np.array(Image.open(IMG_DIR / e["file_name"]).convert("L"))
        p = predict_prob_map(model, device, gray)
        probs[i] = np.array(Image.fromarray((p * 255).astype(np.uint8)).resize(
            (PROB_SIZE, PROB_SIZE), Image.BILINEAR))
    np.save(out_path, probs)
    print(f"saved {out_path}", flush=True)
    return probs


def fuse(kept_rles, prob, t_sem, grow_px, new_min_area):
    """kept_rles in score order (painted, disjoint). Returns fused RLE list."""
    fg = prob > t_sem
    labels = np.zeros((H, W), np.int32)
    for k, r in enumerate(kept_rles, 1):
        labels[mu.decode({"size": [H, W], "counts": r.encode()}).astype(bool)] = k
    inst = labels > 0
    out = list(kept_rles)
    if grow_px > 0 and inst.any():
        region = inst | (fg & (ndimage.distance_transform_edt(~inst) <= grow_px))
        grown = watershed(-prob, markers=labels, mask=region)  # floods fg from each instance, nearest wins
        out = [to_rle((grown == k).astype(np.uint8)) for k in range(1, len(kept_rles) + 1)]
        inst = grown > 0
    if new_min_area:
        comp, n = ndimage.label(fg & ~ndimage.binary_dilation(inst, iterations=3))
        if n:
            sizes = ndimage.sum(np.ones_like(comp), comp, range(1, n + 1))
            out += [to_rle((comp == c).astype(np.uint8)) for c in np.where(sizes >= new_min_area)[0] + 1]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--unet")
    p.add_argument("--probs", default="unet_val_probs.npy")
    p.add_argument("--cache", default="ens_hires_e5.pkl")
    p.add_argument("--sources", nargs="+", default=["A", "B1280", "C"])
    p.add_argument("--accept", type=float, default=0.5)
    p.add_argument("--dedup", type=float, default=0.05)
    p.add_argument("--ts", type=float, nargs="+", help="growth-only grid: U-Net thresholds")
    p.add_argument("--gs", type=int, nargs="+", help="growth-only grid: max grow px")
    args = p.parse_args()

    probs = val_prob_maps(args.unet, args.probs) if args.unet else np.load(args.probs)
    cache = pickle.load(open(args.cache, "rb"))
    cals = fit_calibrators(cache, args.sources, list(range(len(cache))))
    kept_all = []
    for per_source, gt in cache:
        pooled = []
        for key in args.sources:
            if per_source[key]:
                cs = cals[key].predict([s for s, _, _ in per_source[key]])
                pooled += [(float(c), r) for c, (_, r, _) in zip(cs, per_source[key]) if c >= args.accept]
        kept_all.append(paint_panoptic_rle(dedup_nms_rle(pooled, args.dedup)))

    grid = [(0.5, 0, 0)]  # baseline: no fusion
    grid += [(t, g, 0) for t in (0.3, 0.5, 0.7) for g in (15, 40, 100)]
    grid += [(t, 40, a) for t in (0.5, 0.7, 0.85) for a in (300, 1000)]
    if args.ts:
        grid = [(0.5, 0, 0)] + [(t, g, 0) for t in args.ts for g in args.gs]
    totals = {cfg: [0.0, 0.0, 0, 0] for cfg in grid}  # PQ num, den, TP, n_pred
    for i, (kept, (_, gt)) in enumerate(zip(kept_all, cache)):
        prob = np.array(Image.fromarray(probs[i]).resize((W, H), Image.BILINEAR)).astype(np.float32) / 255
        for t, g, a in grid:
            fused = kept if (g == 0 and a == 0) else fuse(kept, prob, t, g, a)
            n_, d_, t_ = pq_against_gt(fused, gt)
            acc = totals[(t, g, a)]
            acc[0] += n_; acc[1] += d_; acc[2] += t_; acc[3] += len(fused)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(cache)} images", flush=True)
    for (t, g, a), (num, den, tp, n_pred) in totals.items():
        print(f"t_sem={t} grow={g}px new_min_area={a}: PQ={num / den:.4f} TP={tp} FP={n_pred - tp} "
              f"SQ={num / max(tp, 1):.4f}", flush=True)


if __name__ == "__main__":
    main()
