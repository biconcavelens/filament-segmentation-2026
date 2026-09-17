"""Inference with the fine-tuned Mask2Former model.

Unlike the Mask R-CNN pipeline, Mask2Former's post-processing already
resolves overlaps internally (via overlap_mask_area_threshold), so no
separate panoptic-painting step is needed.

Usage:
    python predict_mask2former.py --out submission.csv     # full test set
    python predict_mask2former.py --eval-val               # local PQ on val split
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import Mask2FormerImageProcessor
import pycocotools.mask as mu

from train_mask2former import build_model, CHECKPOINT_NAME
from mask2former_dataset import INPUT_SIZE
from predict_trained import to_rle, paint_panoptic

D = Path("data/MAGFiLO_1.0_Kaggle_2026")
TEST_DIR = D / "test" / "test_images"
H, W = 2048, 2048

SCORE_THRESHOLD = 0.85  # picked via evaluate_mask2former.py sweep against real local PQ
MASK_THRESHOLD = 0.5
MIN_AREA = 20
CHUNK_SIZE = 20
COOLDOWN_SECONDS = 15


def load_model(checkpoint: str, device):
    processor = Mask2FormerImageProcessor.from_pretrained(
        CHECKPOINT_NAME, size={"shortest_edge": INPUT_SIZE, "longest_edge": INPUT_SIZE},
        do_reduce_labels=False,
    )
    model = build_model(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, processor


@torch.no_grad()
def predict_one(model, processor, device, pil_img: Image.Image) -> list[np.ndarray]:
    inputs = processor(pil_img, return_tensors="pt").to(device)
    out = model(**inputs)
    result = processor.post_process_instance_segmentation(
        out, threshold=SCORE_THRESHOLD, mask_threshold=MASK_THRESHOLD,
        target_sizes=[(H, W)], return_binary_maps=True,
    )[0]

    masks = result["segmentation"]  # [N,H,W]; HF claims disjoint but validation found overlaps
    if masks.numel() == 0 or masks.ndim < 3:
        return []
    masks = masks.cpu().numpy()
    scores = [seg["score"] for seg in result["segments_info"]]
    candidates = [(scores[i], masks[i].astype(np.uint8)) for i in range(len(masks))]
    return paint_panoptic(candidates, MIN_AREA)


def predict(checkpoint: str, out_csv: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model(checkpoint, device)
    print(f"loaded {checkpoint} on {device}")

    partial_path = Path(out_csv + ".partial")
    files = sorted(TEST_DIR.iterdir())

    done_stems = set()
    rows = []
    if partial_path.exists():
        prev = pd.read_csv(partial_path, dtype=str)
        rows = prev.to_dict("records")
        done_stems = set(prev["filament_id"].str.rsplit("_", n=1).str[0])
        print(f"resuming: {len(done_stems)} images already done")

    remaining = [p for p in files if p.stem not in done_stems]
    for chunk_start in range(0, len(remaining), CHUNK_SIZE):
        chunk = remaining[chunk_start:chunk_start + CHUNK_SIZE]
        for path in chunk:
            pil_img = Image.open(path).convert("RGB")
            kept = predict_one(model, processor, device, pil_img)
            stem = path.stem
            rows.extend({"filament_id": f"{stem}_{k}", "segmentation_rle": to_rle(m)}
                        for k, m in enumerate(kept, 1))
            print(f"{stem}: {len(kept)} kept")

        pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(
            partial_path, index=False)
        done = min(chunk_start + CHUNK_SIZE, len(remaining))
        print(f"--- chunk done: {done}/{len(remaining)} remaining "
              f"({len(files) - len(remaining) + done}/{len(files)} total) ---")
        if done < len(remaining):
            torch.cuda.empty_cache() if device.type == "cuda" else None
            time.sleep(COOLDOWN_SECONDS)

    partial_path.replace(out_csv)
    out_df = pd.read_csv(out_csv)
    print(f"\nwrote {out_csv}: {len(out_df)} rows, {len(files)} images, "
          f"avg {len(out_df)/len(files):.1f} filaments/image")


def eval_val(checkpoint: str, n: int = 30):
    from dataset import train_val_split, IMG_DIR
    from pq import compute_pq

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_model(checkpoint, device)
    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)

    pqs = []
    for i, e in enumerate(val_entries[:n], 1):
        pil_img = Image.open(IMG_DIR / e["file_name"]).convert("RGB")
        kept = predict_one(model, processor, device, pil_img)
        pred_rles = [to_rle(m) for m in kept]

        gt_rles = []
        for a in per_image.get(e["id"], []):
            rles = mu.frPyObjects(a["segmentation"], H, W)
            gt_rles.append(to_rle(mu.decode(mu.merge(rles))))

        pq = compute_pq(pred_rles, gt_rles, H, W)
        pqs.append(pq)
        print(f"[{i}/{n}] {e['file_name']}: pred={len(kept):3d} gt={len(gt_rles):3d} PQ={pq:.3f}")
        if i % CHUNK_SIZE == 0 and i < n:
            torch.cuda.empty_cache()
            time.sleep(COOLDOWN_SECONDS)

    print(f"\nmean PQ over {len(pqs)} val images: {np.mean(pqs):.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/mask2former_best.pt")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--eval-val", action="store_true")
    args = p.parse_args()

    if args.eval_val:
        eval_val(args.checkpoint)
    else:
        predict(args.checkpoint, args.out)
