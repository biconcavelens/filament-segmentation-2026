"""Zero-shot SAM2 automatic mask generation for filament segmentation --
no fine-tuning, no external detector. A single point prompt at a filament's
center scored IoU=0.717 against GT in a smoke test, so the main engineering
problem is getting good candidate POINTS automatically.

Domain-informed grid: filaments are dark absorption features on a bright
disk (confirmed by the MAGFiLO paper and community notebooks), so candidate
points are sampled from dark pixels on a coarse grid rather than a naive
dense grid over the whole disk -- cuts wasted SAM2 calls on obviously-bright
background while still covering everywhere a filament could plausibly be.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage
from transformers import Sam2Model, Sam2Processor
import pycocotools.mask as mu

from dataset import H, W, IMG_DIR, train_val_split
from predict_trained import paint_panoptic, to_rle

D = Path("data/MAGFiLO_1.0_Kaggle_2026")
TEST_DIR = D / "test" / "test_images"

GRID_STEP = 48           # px between candidate points
TOPHAT_SIZE = 21         # structuring element; filaments are locally dark, not globally dark
TOPHAT_THRESHOLD = 15    # ~87% filament-pixel recall at ~34% of disk flagged, measured empirically
IOU_SCORE_THRESH = 0.80  # SAM2's own predicted-IoU confidence gate
MIN_AREA = 20
MAX_AREA_FRAC = 0.01    # drop masks covering an implausibly large fraction of the image (not a filament)
POINT_BATCH = 32        # points per SAM2 forward call
DISK_CENTER = (1024.0, 1024.0)
DISK_RADIUS = 1020.0
CHUNK_SIZE = 10
COOLDOWN_SECONDS = 15


def sample_points(gray: np.ndarray) -> list[tuple[int, int]]:
    """Filaments are locally dark against their immediate surroundings, not
    globally dark against the whole disk (a global percentile threshold
    never catches them -- filament intensity 85-148 vs. the disk's globally
    darkest 20% being <=69, confirmed empirically). Black tophat (closing -
    image) isolates features darker than their local neighborhood instead."""
    yy, xx = np.mgrid[0:H, 0:W]
    disk_mask = (xx - DISK_CENTER[0]) ** 2 + (yy - DISK_CENTER[1]) ** 2 <= DISK_RADIUS ** 2

    gray_f = gray.astype(np.float64)
    closed = ndimage.grey_closing(gray_f, size=(TOPHAT_SIZE, TOPHAT_SIZE))
    tophat = closed - gray_f
    dark = (tophat >= TOPHAT_THRESHOLD) & disk_mask

    gh, gw = H // GRID_STEP, W // GRID_STEP
    points = []
    for gy in range(gh):
        for gx in range(gw):
            y0, y1 = gy * GRID_STEP, (gy + 1) * GRID_STEP
            x0, x1 = gx * GRID_STEP, (gx + 1) * GRID_STEP
            cell = dark[y0:y1, x0:x1]
            if cell.any():
                # the single most-confidently-dark pixel in the cell, not the
                # centroid of all dark pixels (which can drift off a thin,
                # curved filament and land in background between them)
                local_tophat = tophat[y0:y1, x0:x1]
                ys, xs = np.where(cell)
                best = np.argmax(local_tophat[ys, xs])
                points.append((x0 + int(xs[best]), y0 + int(ys[best])))
    return points


@torch.no_grad()
def predict_one(model, processor, device, pil_img: Image.Image, gray: np.ndarray) -> list[np.ndarray]:
    points = sample_points(gray)
    if not points:
        return []

    candidates = []
    for i in range(0, len(points), POINT_BATCH):
        batch = points[i:i + POINT_BATCH]
        input_points = [[[list(p)] for p in batch]]
        input_labels = [[[1] for _ in batch]]
        inputs = processor(pil_img, input_points=input_points, input_labels=input_labels,
                            return_tensors="pt").to(device)
        out = model(**inputs, multimask_output=True)
        scores = out.iou_scores.cpu().numpy()[0]  # [n_points, 3]

        # select the best-scoring mask PER POINT on the low-res (256x256)
        # logits before upsampling -- upsampling all 3 candidates for a full
        # batch to 2048x2048 at once needs several GB of CPU RAM and OOMs
        best_k = scores.argmax(axis=-1)  # [n_points]
        n = len(batch)
        selected = out.pred_masks[0, torch.arange(n), torch.from_numpy(best_k).to(device)]
        selected = selected.unsqueeze(1).unsqueeze(0).cpu()  # [1, n_points, 1, 256, 256]
        masks = processor.post_process_masks(selected, inputs["original_sizes"].cpu())[0]  # [n_points,1,H,W]

        for j in range(n):
            score = float(scores[j, best_k[j]])
            if score < IOU_SCORE_THRESH:
                continue
            m = masks[j, 0].numpy().astype(np.uint8)
            area = int(m.sum())
            if area < MIN_AREA or area > MAX_AREA_FRAC * H * W:
                continue
            candidates.append((score, m))

    return paint_panoptic(candidates, MIN_AREA)


def load_models(device):
    model = Sam2Model.from_pretrained("facebook/sam2-hiera-base-plus").to(device)
    processor = Sam2Processor.from_pretrained("facebook/sam2-hiera-base-plus")
    model.eval()
    return model, processor


def eval_val(n: int = 20):
    from pq import compute_pq
    device = torch.device("cuda")
    model, processor = load_models(device)
    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)

    pqs = []
    t0 = time.time()
    for i, e in enumerate(val_entries[:n], 1):
        pil_img = Image.open(IMG_DIR / e["file_name"]).convert("RGB")
        gray = np.array(pil_img.convert("L"))
        kept = predict_one(model, processor, device, pil_img, gray)
        pred_rles = [to_rle(m) for m in kept]

        gt_rles = []
        for a in per_image.get(e["id"], []):
            rles = mu.frPyObjects(a["segmentation"], H, W)
            gt_rles.append(to_rle(mu.decode(mu.merge(rles))))

        pq = compute_pq(pred_rles, gt_rles, H, W)
        pqs.append(pq)
        print(f"[{i}/{n}] {e['file_name']}: pred={len(kept):3d} gt={len(gt_rles):3d} "
              f"PQ={pq:.3f} ({time.time()-t0:.0f}s elapsed)", flush=True)

    print(f"\nmean PQ over {len(pqs)} val images: {np.mean(pqs):.3f}")


def predict(out_csv: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_models(device)
    print(f"loaded SAM2 on {device}")

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
            gray = np.array(pil_img.convert("L"))
            kept = predict_one(model, processor, device, pil_img, gray)
            stem = path.stem
            rows.extend({"filament_id": f"{stem}_{k}", "segmentation_rle": to_rle(m)}
                        for k, m in enumerate(kept, 1))
            print(f"{stem}: {len(kept)} kept", flush=True)

        pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(
            partial_path, index=False)
        done = min(chunk_start + CHUNK_SIZE, len(remaining))
        print(f"--- chunk done: {done}/{len(remaining)} remaining "
              f"({len(files) - len(remaining) + done}/{len(files)} total) ---", flush=True)
        if done < len(remaining):
            torch.cuda.empty_cache() if device.type == "cuda" else None
            time.sleep(COOLDOWN_SECONDS)

    partial_path.replace(out_csv)
    out_df = pd.read_csv(out_csv)
    print(f"\nwrote {out_csv}: {len(out_df)} rows, {len(files)} images, "
          f"avg {len(out_df)/len(files):.1f} filaments/image")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.out:
        predict(args.out)
    else:
        eval_val(args.n)
