"""Can we complete truncated filament masks? diag_near_miss_shape.py found
Experiment A's ~97 near misses (IoU 0.3-0.5) are mostly truncated -- median
precision 0.78, recall 0.49, the GT extending past our mask into a fainter
tail. Here each kept mask is re-refined with a crop re-centred on the
*refined* mask at a wider context, optionally unioned with the original
(grow-only), for a few iterations; then dedup + paint again and score PQ.

CPU-only (the refiner is small); uses the cached val candidates.
    python sweep_grow.py
"""
import pickle

import numpy as np
import pycocotools.mask as mu
import torch
from PIL import Image

from crop_dataset import square_bounds, CROP_SIZE
from dataset import train_val_split, IMG_DIR
from predict_refined import refine_with_tta
from predict_trained import to_rle
from sweep_ensemble_4way import (CACHE_PATH, H, W, REFINER_CKPT, fit_calibrators,
                                 dedup_nms_rle, paint_panoptic_rle, pq_against_gt)
from train_refiner import load_refiner

# the real-0.39 2-way pipeline (YOLO@2048 retired: every submission including it scored 0.38)
SOURCES, ACCEPT, DEDUP = ["A", "B1280"], 0.45, 0.05
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIGS = [  # (context, union_with_previous, iterations)
    (1.8, False, 1), (2.4, False, 1), (3.0, False, 1),
    (2.4, True, 1), (2.4, True, 2), (3.0, True, 2),
]


@torch.no_grad()
def refine(refiner, gray, mask, context, thresholds=(0.5,)):
    r0 = refiner[0] if isinstance(refiner, list) else refiner
    x0, y0, x1, y1 = square_bounds(mask.astype(bool), context=context)
    crop = np.array(Image.fromarray(gray[y0:y1, x0:x1]).resize(
        (r0.crop_size, r0.crop_size), Image.BILINEAR)).astype(np.float32) / 255.0
    channels = [crop]
    if r0.in_channels == 2:  # hint refiner: the mask being re-refined is the hint
        channels.append((np.array(Image.fromarray(mask[y0:y1, x0:x1].astype(np.uint8) * 255).resize(
            (r0.crop_size, r0.crop_size), Image.NEAREST)) > 127).astype(np.float32))
    refiners = refiner if isinstance(refiner, list) else [refiner]  # several: average their probabilities
    prob = np.mean([refine_with_tta(r, DEVICE, np.stack(channels)) for r in refiners], axis=0)
    side = y1 - y0
    prob_full = np.array(Image.fromarray((prob * 255).astype(np.uint8)).resize(
        (side, side), Image.BILINEAR)).astype(np.float32) / 255.0
    outs = []
    for t in thresholds:
        out = np.zeros((H, W), dtype=np.uint8)
        out[y0:y1, x0:x1] = prob_full > t
        outs.append(out)
    return outs[0] if len(outs) == 1 else outs


def main():
    global CONFIGS, DEVICE
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--refiner", default=REFINER_CKPT, help="comma list averages several refiners")
    p.add_argument("--base-only", action="store_true",
                   help="only re-refine at the normal context (compare refiners like-for-like)")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--thresholds", default=None,
                   help="comma list: re-refine once at context 1.8 and cut the refiner probability at each")
    args = p.parse_args()
    thresholds = [float(t) for t in args.thresholds.split(",")] if args.thresholds else None
    if thresholds:
        CONFIGS = [(1.8, False, t) for t in thresholds]  # 3rd field reused as the threshold
    if args.base_only:
        CONFIGS = [(1.8, False, 1)]
    if args.cpu:
        DEVICE = torch.device("cpu")
    print(f"refiner={args.refiner} device={DEVICE}", flush=True)

    torch.set_num_threads(8)
    refiner = [load_refiner(r, DEVICE) for r in args.refiner.split(",")]

    cache = pickle.load(open(CACHE_PATH, "rb"))
    cals = fit_calibrators(cache, SOURCES, list(range(len(cache))))
    _, val_entries, _ = train_val_split(val_frac=0.1, seed=0)

    totals = {cfg: [0.0, 0.0, 0] for cfg in CONFIGS}
    base = [0.0, 0.0, 0]
    for n_img, ((per_source, gt), e) in enumerate(zip(cache, val_entries), 1):
        pooled = []
        for key in SOURCES:
            if per_source[key]:
                cs = cals[key].predict([s for s, _, _ in per_source[key]])
                pooled += [(float(c), r) for c, (_, r, _) in zip(cs, per_source[key]) if c >= ACCEPT]
        deduped = dedup_nms_rle(pooled, DEDUP)
        for acc, res in [(base, pq_against_gt(paint_panoptic_rle(deduped), gt))]:
            acc[0] += res[0]; acc[1] += res[1]; acc[2] += res[2]
        if not deduped:
            for cfg in CONFIGS:
                res = pq_against_gt([], gt)
                totals[cfg][1] += res[1]
            continue
        gray = np.array(Image.open(IMG_DIR / e["file_name"]).convert("L"))
        masks = [(s, mu.decode({"size": [H, W], "counts": r.encode()})) for s, r in deduped]
        if thresholds:
            per_t = [[] for _ in thresholds]
            for s, m in masks:
                for lst, new in zip(per_t, refine(refiner, gray, m, 1.8, thresholds + [0.5])[:len(thresholds)]):
                    if new.sum():
                        lst.append((s, to_rle(new)))
            for cfg, grown in zip(CONFIGS, per_t):
                res = pq_against_gt(paint_panoptic_rle(dedup_nms_rle(grown, DEDUP)), gt)
                totals[cfg][0] += res[0]; totals[cfg][1] += res[1]; totals[cfg][2] += res[2]
            if n_img % 20 == 0:
                print(f"  {n_img}/{len(cache)} images", flush=True)
            continue
        for cfg in CONFIGS:
            context, union, iters = cfg
            grown = []
            for s, m in masks:
                cur = m
                for _ in range(iters):
                    new = refine(refiner, gray, cur, context)
                    if new.sum() == 0:
                        break
                    cur = (cur | new) if union else new
                grown.append((s, to_rle(cur)))
            res = pq_against_gt(paint_panoptic_rle(dedup_nms_rle(grown, DEDUP)), gt)
            totals[cfg][0] += res[0]; totals[cfg][1] += res[1]; totals[cfg][2] += res[2]
        if n_img % 20 == 0:
            print(f"  {n_img}/{len(cache)} images", flush=True)

    print(f"baseline (cached refine): PQ={base[0] / base[1]:.4f} TP={base[2]}")
    for cfg in CONFIGS:
        num, den, tp = totals[cfg]
        print(f"context={cfg[0]} union={cfg[1]} iters={cfg[2]}: PQ={num / den:.4f} TP={tp}", flush=True)


if __name__ == "__main__":
    main()
