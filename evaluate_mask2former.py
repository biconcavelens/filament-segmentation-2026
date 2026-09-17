"""Sweep score/mask thresholds for Mask2Former against real local PQ.

Caches raw model outputs once per val image (expensive, GPU), then sweeps
thresholds against the cache (cheap, no GPU).
"""
import time

import numpy as np
import torch
import pycocotools.mask as mu

from dataset import train_val_split, IMG_DIR, H, W
from predict_mask2former import load_model, to_rle
from pq import compute_pq
from PIL import Image

CHECKPOINT = "checkpoints/mask2former_best.pt"
SCORE_THRESHOLDS = [0.85, 0.9, 0.93, 0.95, 0.97]
MASK_THRESHOLD = 0.5
MIN_AREA = 20
CHUNK_SIZE = 20
COOLDOWN_SECONDS = 10


@torch.no_grad()
def cache_outputs(model, processor, device, val_entries, per_image, n):
    cache = []
    for i, e in enumerate(val_entries[:n], 1):
        pil_img = Image.open(IMG_DIR / e["file_name"]).convert("RGB")
        inputs = processor(pil_img, return_tensors="pt").to(device)
        out = model(**inputs)

        gt_rles = []
        for a in per_image.get(e["id"], []):
            rles = mu.frPyObjects(a["segmentation"], H, W)
            gt_rles.append(to_rle(mu.decode(mu.merge(rles))))

        cache.append((out, gt_rles))
        if i % CHUNK_SIZE == 0 and i < n:
            torch.cuda.empty_cache()
            time.sleep(COOLDOWN_SECONDS)
    return cache


def sweep(cache, processor):
    for st in SCORE_THRESHOLDS:
        pqs, npred = [], []
        for out, gt_rles in cache:
            result = processor.post_process_instance_segmentation(
                out, threshold=st, mask_threshold=MASK_THRESHOLD,
                target_sizes=[(H, W)], return_binary_maps=True,
            )[0]
            masks = result["segmentation"]
            if masks.numel() == 0 or masks.ndim < 3:
                pred_rles = []
            else:
                masks = masks.cpu().numpy()
                pred_rles = [to_rle(m.astype(np.uint8)) for m in masks if m.sum() >= MIN_AREA]
            pqs.append(compute_pq(pred_rles, gt_rles, H, W))
            npred.append(len(pred_rles))
        print(f"score>={st:.1f}  mean_PQ={np.mean(pqs):.3f}  mean_pred={np.mean(npred):5.1f}",
              flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model(CHECKPOINT, device)
    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)

    t0 = time.time()
    cache = cache_outputs(model, processor, device, val_entries, per_image, n=30)
    print(f"cached in {time.time()-t0:.0f}s", flush=True)
    sweep(cache, processor)


if __name__ == "__main__":
    main()
