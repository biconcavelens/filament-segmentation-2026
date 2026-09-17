"""Sweep binarization threshold x min_area for the U-Net against real local PQ.

Caches per-image probability maps once (expensive, GPU), then sweeps
threshold/min_area against the cache (cheap, CPU-only).
"""
import time

import numpy as np
import torch
import pycocotools.mask as mu
from scipy import ndimage

from dataset import train_val_split, IMG_DIR, H, W
from predict_unet import load_model, predict_prob_map, to_rle
from pq import compute_pq
from PIL import Image

CHECKPOINT = "checkpoints/unet_best.pt"
THRESHOLDS = [0.75, 0.8, 0.85, 0.9]
MIN_AREAS = [150, 200, 300, 400]
CHUNK_SIZE = 20
COOLDOWN_SECONDS = 8


@torch.no_grad()
def cache_probs(model, device, val_entries, per_image, n):
    cache = []
    for i, e in enumerate(val_entries[:n], 1):
        gray = np.array(Image.open(IMG_DIR / e["file_name"]).convert("L"))
        prob = predict_prob_map(model, device, gray)

        gt_rles = []
        for a in per_image.get(e["id"], []):
            rles = mu.frPyObjects(a["segmentation"], H, W)
            gt_rles.append(to_rle(mu.decode(mu.merge(rles))))

        cache.append((prob, gt_rles))
        if i % CHUNK_SIZE == 0 and i < n:
            torch.cuda.empty_cache()
            time.sleep(COOLDOWN_SECONDS)
    return cache


def sweep(cache):
    best = None
    for th in THRESHOLDS:
        for min_area in MIN_AREAS:
            pqs, npred = [], []
            for prob, gt_rles in cache:
                binary = (prob > th).astype(np.uint8)
                labeled, n = ndimage.label(binary, structure=np.ones((3, 3)))
                if n == 0:
                    pred_rles = []
                else:
                    sizes = ndimage.sum(binary, labeled, range(1, n + 1))
                    keep_ids = np.where(sizes >= min_area)[0] + 1
                    pred_rles = [to_rle((labeled == i).astype(np.uint8)) for i in keep_ids]
                pqs.append(compute_pq(pred_rles, gt_rles, H, W))
                npred.append(len(pred_rles))
            md = np.mean(pqs)
            if best is None or md > best[0]:
                best = (md, th, min_area, np.mean(npred))
            print(f"threshold={th:.1f} min_area={min_area:4d}  mean_PQ={md:.3f}  "
                  f"mean_pred={np.mean(npred):5.1f}", flush=True)
    print("\nBEST:", best)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CHECKPOINT, device)
    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)

    t0 = time.time()
    cache = cache_probs(model, device, val_entries, per_image, n=30)
    print(f"cached in {time.time()-t0:.0f}s", flush=True)
    sweep(cache)


if __name__ == "__main__":
    main()
