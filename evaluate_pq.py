"""Sweep checkpoint x score_threshold against real local PQ on val set.

Runs inference once per checkpoint (expensive, GPU), caches raw (score,
mask-at-0.5) candidates, then sweeps score thresholds against the cache
(cheap, no GPU) so we're not blindly guessing which epoch/threshold to ship.
"""
import time
from pathlib import Path

import numpy as np
import torch
import pycocotools.mask as mu

from dataset import train_val_split, IMG_DIR, H, W
from train import build_model
from predict_trained import paint_panoptic, to_rle
from pq import compute_pq
from PIL import Image

CHECKPOINTS = ["checkpoints/maskrcnn_epoch3.pt", "checkpoints/maskrcnn_epoch5.pt",
               "checkpoints/maskrcnn_epoch7.pt"]
SCORE_THRESHOLDS = [0.75, 0.8, 0.85, 0.9, 0.95]
CACHE_SCORE_FLOOR = 0.7  # only need candidates above the previous sweep's best
COOLDOWN_SECONDS = 10
CHUNK_SIZE = 20


@torch.no_grad()
def cache_candidates(checkpoint_path: str, val_entries: list[dict], per_image: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=2).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    cache = []  # (candidates:[(score, rle)], gt_rles:[str])
    for i, e in enumerate(val_entries, 1):
        img = Image.open(IMG_DIR / e["file_name"]).convert("RGB")
        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        out = model([img_t.to(device)])[0]
        scores = out["scores"].cpu().numpy()
        masks = out["masks"].cpu().numpy()

        candidates = []
        for j in range(len(scores)):
            if scores[j] < CACHE_SCORE_FLOOR:
                continue
            binary = (masks[j, 0] > 0.5).astype(np.uint8)
            if binary.sum() == 0:
                continue
            candidates.append((float(scores[j]), to_rle(binary)))

        gt_rles = []
        for a in per_image.get(e["id"], []):
            rles = mu.frPyObjects(a["segmentation"], H, W)
            m = mu.decode(mu.merge(rles))
            gt_rles.append(to_rle(m))

        cache.append((candidates, gt_rles))
        if i % CHUNK_SIZE == 0 and i < len(val_entries):
            torch.cuda.empty_cache()
            time.sleep(COOLDOWN_SECONDS)

    del model
    torch.cuda.empty_cache()
    return cache


def sweep_thresholds(cache, label):
    for st in SCORE_THRESHOLDS:
        pqs, npred = [], []
        for candidates, gt_rles in cache:
            filtered = [(s, r) for s, r in candidates if s >= st]
            filtered_masks = [(s, mu.decode({"size": [H, W], "counts": r.encode("utf-8")}))
                               for s, r in filtered]
            kept = paint_panoptic(filtered_masks, min_area=20)
            pred_rles = [to_rle(m) for m in kept]
            pqs.append(compute_pq(pred_rles, gt_rles, H, W))
            npred.append(len(pred_rles))
        print(f"{label:35s} score>={st:.2f}  mean_PQ={np.mean(pqs):.3f}  "
              f"mean_pred={np.mean(npred):5.1f}", flush=True)


def main():
    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)
    print(f"evaluating on {len(val_entries)} val entries\n")

    for ckpt in CHECKPOINTS:
        if not Path(ckpt).exists():
            print(f"skip {ckpt}: not found")
            continue
        t0 = time.time()
        cache = cache_candidates(ckpt, val_entries, per_image)
        print(f"[{ckpt}] cached in {time.time()-t0:.0f}s", flush=True)
        sweep_thresholds(cache, Path(ckpt).stem)
        print()


if __name__ == "__main__":
    main()
