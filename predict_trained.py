"""Run the fine-tuned Mask R-CNN on test images -> submission.csv.

Mask R-CNN's per-instance masks can overlap (no mask-level NMS by default),
but PQ requires pixel-disjoint predictions per image. Fixed the same way as
the discussion board's score-ordered panoptic painting: sort by confidence,
let each detection claim only pixels not already claimed by a higher-scoring
one, drop what's left if it falls below min_area.

Usage:
    python predict_trained.py --checkpoint checkpoints/maskrcnn_epoch1.pt
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
import pycocotools.mask as mu

from train import build_model, MIN_SIZE, MAX_SIZE

D = Path("data/MAGFiLO_1.0_Kaggle_2026")
TEST_DIR = D / "test" / "test_images"
H, W = 2048, 2048

SCORE_THRESHOLD = 0.8  # picked via evaluate_pq.py sweep against real local PQ
MASK_THRESHOLD = 0.5
MIN_AREA = 20  # post-paint; low since painting already prunes overlap junk

# ponytail: chunking + a cooldown pause exist only because this laptop's GPU
# hard-crashed twice under sustained load; a desktop/cloud GPU wouldn't need
# either. Remove if running somewhere thermally sane.
CHUNK_SIZE = 20
COOLDOWN_SECONDS = 15


def paint_panoptic(candidates: list[tuple[float, np.ndarray]], min_area: int):
    """candidates: list of (score, binary_mask uint8 [H,W]). Returns
    non-overlapping masks, highest-confidence detection wins each pixel."""
    candidates = sorted(candidates, key=lambda x: -x[0])
    claimed = np.zeros((H, W), dtype=np.uint8)
    kept = []
    for score, binary in candidates:
        remaining = binary & (1 - claimed)
        area = int(remaining.sum())
        if area < min_area:
            continue
        claimed |= remaining
        kept.append(remaining)
    return kept


def to_rle(mask: np.ndarray) -> str:
    return mu.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("utf-8")


@torch.no_grad()
def predict_one(model, device, path: Path) -> list[dict]:
    img = Image.open(path).convert("RGB")
    img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    out = model([img_t.to(device)])[0]

    scores = out["scores"].cpu().numpy()
    masks = out["masks"].cpu().numpy()  # [N,1,H,W] soft, already resized to input res

    candidates = []
    for j in range(len(scores)):
        if scores[j] < SCORE_THRESHOLD:
            continue
        binary = (masks[j, 0] > MASK_THRESHOLD).astype(np.uint8)
        if binary.sum() == 0:
            continue
        candidates.append((float(scores[j]), binary))

    kept = paint_panoptic(candidates, MIN_AREA)
    stem = path.stem
    print(f"{stem}: {len(candidates)} raw -> {len(kept)} kept")
    return [{"filament_id": f"{stem}_{k}", "segmentation_rle": to_rle(m)}
            for k, m in enumerate(kept, 1)]


def predict(checkpoint_path: str, out_csv: str = "submission.csv"):
    """Chunked + resumable: writes progress to `<out_csv>.partial` after every
    CHUNK_SIZE images and pauses between chunks, so a crash loses at most one
    chunk instead of the whole run. Re-run the same command to resume."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=2).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"loaded {checkpoint_path} (epoch {state['epoch']}) on {device}")

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
            rows.extend(predict_one(model, device, path))

        pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(
            partial_path, index=False
        )
        done = min(chunk_start + CHUNK_SIZE, len(remaining))
        print(f"--- chunk done: {done}/{len(remaining)} remaining images "
              f"({len(files) - len(remaining) + done}/{len(files)} total) ---")

        if done < len(remaining):
            torch.cuda.empty_cache() if device.type == "cuda" else None
            time.sleep(COOLDOWN_SECONDS)

    partial_path.replace(out_csv)  # replace() overwrites on Windows; rename() doesn't
    out_df = pd.read_csv(out_csv)
    print(f"\nwrote {out_csv}: {len(out_df)} rows, {len(files)} images, "
          f"avg {len(out_df)/len(files):.1f} filaments/image")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="submission.csv")
    args = p.parse_args()
    predict(args.checkpoint, args.out)
