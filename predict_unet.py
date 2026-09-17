"""U-Net inference: predict at train resolution, upsample, threshold,
split into instances via connected components (same recipe as the
community's real 0.57 notebook).

Usage:
    python predict_unet.py --out submission.csv     # full test set
    python predict_unet.py --eval-val               # local PQ on val split
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage
import pycocotools.mask as mu

from train_unet import build_model
from unet_dataset import TRAIN_SIZE, val_transform
from predict_trained import to_rle

D = Path("data/MAGFiLO_1.0_Kaggle_2026")
TEST_DIR = D / "test" / "test_images"
H, W = 2048, 2048

THRESHOLD = 0.8  # picked via evaluate_unet.py sweep against real local PQ
MIN_AREA = 300
CHUNK_SIZE = 20
COOLDOWN_SECONDS = 10


def load_model(checkpoint: str, device):
    model = build_model().to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


@torch.no_grad()
def predict_prob_map(model, device, gray_img: np.ndarray) -> np.ndarray:
    """Returns a full-resolution (H,W) probability map in [0,1]."""
    out = val_transform(image=gray_img, mask=np.zeros_like(gray_img, dtype=np.float32))
    img_t = out["image"].unsqueeze(0).to(device)
    logits = model(img_t)
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    probs_full = np.array(Image.fromarray((probs * 255).astype(np.uint8)).resize(
        (W, H), Image.BILINEAR)).astype(np.float32) / 255.0
    return probs_full


def masks_from_prob(prob_map: np.ndarray, threshold: float = THRESHOLD,
                     min_area: int = MIN_AREA) -> list[np.ndarray]:
    binary = (prob_map > threshold).astype(np.uint8)
    labeled, n = ndimage.label(binary, structure=np.ones((3, 3)))
    if n == 0:
        return []
    sizes = ndimage.sum(binary, labeled, range(1, n + 1))
    keep_ids = np.where(sizes >= min_area)[0] + 1
    return [(labeled == i).astype(np.uint8) for i in keep_ids]


def predict(checkpoint: str, out_csv: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, device)
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
            gray = np.array(Image.open(path).convert("L"))
            prob = predict_prob_map(model, device, gray)
            kept = masks_from_prob(prob)
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
    model = load_model(checkpoint, device)
    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)

    pqs = []
    for i, e in enumerate(val_entries[:n], 1):
        gray = np.array(Image.open(IMG_DIR / e["file_name"]).convert("L"))
        prob = predict_prob_map(model, device, gray)
        kept = masks_from_prob(prob)
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
    p.add_argument("--checkpoint", default="checkpoints/unet_best.pt")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--eval-val", action="store_true")
    args = p.parse_args()

    if args.eval_val:
        eval_val(args.checkpoint)
    else:
        predict(args.checkpoint, args.out)
