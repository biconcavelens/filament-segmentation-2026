"""Watershed instance decoding for the dense semantic+distance model.

Naive per-pixel local-maxima watershed over-fragments long thin filaments
(a single filament's distance ridge is fairly flat along its length, so
noise alone can create several spurious peaks). Mitigation: connected-
component the semantic mask first: normal-sized blobs (one filament) are
kept whole; only blobs large enough to plausibly be several touching
filaments get watershed-split, using well-separated, prominent peaks only.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage
from skimage.segmentation import watershed

from dataset import H, W, IMG_DIR, train_val_split
from watershed_dataset import PROC_SIZE
from train_watershed import WatershedUNet
from predict_trained import to_rle

D = Path("data/MAGFiLO_1.0_Kaggle_2026")
TEST_DIR = D / "test" / "test_images"

SEMANTIC_THRESHOLD = 0.5
MIN_AREA = 20
CHUNK_SIZE = 20
COOLDOWN_SECONDS = 15
# a single Left/Right-chiral filament averages ~2400px segmentation area at
# native 2048 resolution (MAGFiLO paper stats); a blob much bigger than that
# is a plausible merge of 2+ touching filaments worth trying to split
SPLIT_AREA_THRESHOLD = 4500
CORE_FRACTION = 0.55       # marker seeds = pixels above this fraction of the blob's own max distance
MIN_MARKER_AREA = 15       # drop tiny marker fragments (noise, not a real second instance)


@torch.no_grad()
def predict_maps(model, device, gray_2048: np.ndarray):
    """Run the model, return (semantic_prob, distance) upsampled to native 2048x2048."""
    small = np.array(Image.fromarray(gray_2048).resize((PROC_SIZE, PROC_SIZE), Image.BILINEAR))
    x = torch.from_numpy(small).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
    logits = model(x)[0].cpu().numpy()
    sem_prob = 1 / (1 + np.exp(-logits[0]))
    dist_prob = 1 / (1 + np.exp(-logits[1]))

    sem_full = np.array(Image.fromarray((sem_prob * 255).astype(np.uint8)).resize(
        (W, H), Image.BILINEAR)).astype(np.float32) / 255.0
    dist_full = np.array(Image.fromarray((dist_prob * 255).astype(np.uint8)).resize(
        (W, H), Image.BILINEAR)).astype(np.float32) / 255.0
    return sem_full, dist_full


def instances_from_maps(sem_full: np.ndarray, dist_full: np.ndarray) -> list[np.ndarray]:
    fg = sem_full > SEMANTIC_THRESHOLD
    labels, n = ndimage.label(fg)
    kept = []
    for lbl in range(1, n + 1):
        blob = labels == lbl
        area = int(blob.sum())
        if area < MIN_AREA:
            continue
        if area <= SPLIT_AREA_THRESHOLD:
            kept.append(blob)
            continue

        # candidate merge of touching filaments: region-based markers, not point
        # peaks -- elongated filaments have flat ridges along most of their
        # length, so two touching ones may never form distinct peak POINTS, but
        # their high-confidence CORE REGIONS are still spatially disjoint,
        # separated by the valley at the shared boundary.
        blob_dist = np.where(blob, dist_full, 0)
        core = blob_dist > (CORE_FRACTION * blob_dist.max())
        marker_labels, n_markers = ndimage.label(core)
        sizes = ndimage.sum(core, marker_labels, range(1, n_markers + 1))
        markers = np.where(np.isin(marker_labels, np.where(sizes >= MIN_MARKER_AREA)[0] + 1),
                            marker_labels, 0)
        n_valid = len(np.unique(markers)) - (1 if 0 in markers else 0)
        if n_valid < 2:
            kept.append(blob)  # no clean multi-core split found, keep as one instance
            continue

        split = watershed(-blob_dist, markers=markers, mask=blob)
        for lbl2 in np.unique(split):
            if lbl2 == 0:
                continue
            piece = split == lbl2
            if piece.sum() >= MIN_AREA:
                kept.append(piece)

    return kept


def predict_one(model, device, gray_2048: np.ndarray) -> list[np.ndarray]:
    sem_full, dist_full = predict_maps(model, device, gray_2048)
    return instances_from_maps(sem_full, dist_full)


def eval_val(ckpt: str, n: int = 116):
    from pq import compute_pq
    device = torch.device("cuda")
    model = WatershedUNet().to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"]); model.eval()

    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)
    import pycocotools.mask as mu

    tp = fp_near = fp_spur = fn_near = fn_total = 0
    tp_ious = []
    pq_num = pq_den = 0.0
    for i, e in enumerate(val_entries[:n], 1):
        gray = np.array(Image.open(IMG_DIR / e["file_name"]).convert("L"))
        kept = predict_one(model, device, gray)
        pred = [{"size": [H, W], "counts": to_rle(m).encode()} for m in kept]

        gt = []
        for a in per_image.get(e["id"], []):
            rles = mu.frPyObjects(a["segmentation"], H, W)
            gt.append({"size": [H, W], "counts": to_rle(mu.decode(mu.merge(rles))).encode()})

        if pred and gt:
            iou = mu.iou(pred, gt, [0] * len(gt))
            best_per_gt = iou.max(axis=0); best_per_pred = iou.max(axis=1)
        else:
            best_per_gt = np.zeros(len(gt)); best_per_pred = np.zeros(len(pred))

        m_tp = int((best_per_gt > 0.5).sum())
        tp += m_tp
        tp_ious.extend(best_per_gt[best_per_gt > 0.5].tolist())
        fn_near += int(((best_per_gt > 0.2) & (best_per_gt <= 0.5)).sum())
        fn_total += int((best_per_gt <= 0.2).sum())
        fp_near += int(((best_per_pred > 0.2) & (best_per_pred <= 0.5)).sum())
        fp_spur += int((best_per_pred <= 0.2).sum())

        n_fp = len(pred) - m_tp
        n_fn = len(gt) - m_tp
        pq_num += float(best_per_gt[best_per_gt > 0.5].sum())
        pq_den += m_tp + 0.5 * n_fp + 0.5 * n_fn
        if i % 10 == 0:
            print(f"  {i}/{n}", flush=True)

    n_gt = tp + fn_near + fn_total
    n_pred = tp + fp_near + fp_spur
    print(f"\n=== {n} val entries, {n_gt} GT filaments, {n_pred} predictions (WATERSHED) ===")
    print(f"TP            : {tp:4d}  ({100*tp/n_gt:.0f}% of GT)   mean IoU of TPs = {np.mean(tp_ious):.3f}")
    print(f"FN near-miss  : {fn_near:4d}  ({100*fn_near/n_gt:.0f}% of GT)")
    print(f"FN total-miss : {fn_total:4d}  ({100*fn_total/n_gt:.0f}% of GT)")
    print(f"FP near-miss  : {fp_near:4d}  ({100*fp_near/n_pred:.0f}% of preds)")
    print(f"FP spurious   : {fp_spur:4d}  ({100*fp_spur/n_pred:.0f}% of preds)")
    print(f"aggregate PQ  : {pq_num/pq_den:.3f}")


def predict(ckpt: str, out_csv: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WatershedUNet().to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"]); model.eval()
    print(f"loaded {ckpt} on {device}")

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
            kept = predict_one(model, device, gray)
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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/watershed_best.pt")
    p.add_argument("--n", type=int, default=116)
    p.add_argument("--out", default=None)
    p.add_argument("--split-area", type=float, default=None,
                    help="override SPLIT_AREA_THRESHOLD (100000 effectively disables splitting)")
    args = p.parse_args()
    if args.split_area is not None:
        SPLIT_AREA_THRESHOLD = args.split_area  # module global, read by instances_from_maps
    if args.out:
        predict(args.ckpt, args.out)
    else:
        eval_val(args.ckpt, args.n)
