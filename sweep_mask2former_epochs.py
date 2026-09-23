"""Real PQ evaluation across multiple Mask2Former checkpoints (not just the
lowest-val_loss one) on the FULL 116-image val split, with an extended
score-threshold range. The original evaluate_mask2former.py only tested
score thresholds 0.85-0.97 on 30 val images -- given every box-based
detector this session had severe confidence-miscalibration requiring much
lower thresholds than naively expected, and val_loss has repeatedly not
tracked PQ well this session, this checks a wider net before trusting the
auto-selected "best" checkpoint (lowest val_loss, epoch 16 here).
"""
import time
from types import SimpleNamespace

import numpy as np
import torch
import pycocotools.mask as mu
from PIL import Image

from dataset import train_val_split, IMG_DIR, H, W
from predict_mask2former import load_model, to_rle
from pq import compute_pq

EPOCHS_TO_TEST = [5, 8, 10, 12, 14, 15, 16, 17, 18, 19]
SCORE_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97]
MASK_THRESHOLD = 0.5
MIN_AREA = 20
COOLDOWN_EVERY = 20
COOLDOWN_SECONDS = 5


@torch.no_grad()
def cache_outputs(model, processor, device, val_entries, per_image):
    cache = []
    for i, e in enumerate(val_entries, 1):
        pil_img = Image.open(IMG_DIR / e["file_name"]).convert("RGB")
        inputs = processor(pil_img, return_tensors="pt").to(device)
        out = model(**inputs)
        # only keep what post_process_instance_segmentation actually reads --
        # the full ModelOutput also carries auxiliary per-decoder-layer
        # logits (~9x the memory, used only for training loss), which OOMs
        # a 16GB GPU once 116 images' worth are cached simultaneously
        trimmed = SimpleNamespace(
            class_queries_logits=out.class_queries_logits,
            masks_queries_logits=out.masks_queries_logits,
        )

        gt_rles = []
        for a in per_image.get(e["id"], []):
            rles = mu.frPyObjects(a["segmentation"], H, W)
            gt_rles.append(to_rle(mu.decode(mu.merge(rles))))

        cache.append((trimmed, gt_rles))
        if i % COOLDOWN_EVERY == 0:
            print(f"  cached {i}/{len(val_entries)}", flush=True)
            torch.cuda.empty_cache()
            time.sleep(COOLDOWN_SECONDS)
    return cache


def sweep(cache, processor, epoch):
    results = []
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
        mean_pq = float(np.mean(pqs))
        print(f"epoch={epoch} score>={st:.2f}  mean_PQ={mean_pq:.4f}  mean_pred={np.mean(npred):5.1f}",
              flush=True)
        results.append((mean_pq, epoch, st))
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)
    print(f"evaluating on all {len(val_entries)} val images", flush=True)

    all_results = []
    for epoch in EPOCHS_TO_TEST:
        ckpt = f"checkpoints/mask2former_epoch{epoch}.pt"
        print(f"\n=== loading {ckpt} ===", flush=True)
        model, processor = load_model(ckpt, device)

        t0 = time.time()
        cache = cache_outputs(model, processor, device, val_entries, per_image)
        print(f"cached in {time.time()-t0:.0f}s", flush=True)

        all_results.extend(sweep(cache, processor, epoch))
        del model, cache
        torch.cuda.empty_cache()

    all_results.sort(reverse=True)
    print("\n=== top 15 across all epochs/thresholds ===")
    for pq, epoch, st in all_results[:15]:
        print(f"PQ={pq:.4f}  epoch={epoch}  score>={st:.2f}")


if __name__ == "__main__":
    main()
