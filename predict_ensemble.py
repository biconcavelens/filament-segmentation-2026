"""Recall-boosting ensemble: Mask R-CNN detector boxes + U-Net-segmentation
connected-component boxes, both refined through the same crop-refiner U-Net.

Rationale: a missed detection is a permanent false negative that PQ punishes
hard. The detector and the independently-trained U-Net almost certainly miss
different filaments, so their union should have meaningfully higher recall
than either alone -- and every box, regardless of source, still gets cleaned
up by the same proven refiner + TTA before scoring.

Usage:
    python predict_ensemble.py --out submission.csv     # full test set
    python predict_ensemble.py --eval-val               # local PQ on val split
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

from train import build_model as build_detector
from train_refiner import RefinerUNet
from train_unet import build_model as build_unet
from unet_dataset import val_transform
from crop_dataset import square_bounds, CROP_SIZE
from predict_trained import paint_panoptic, to_rle
from predict_refined import refine_with_tta

D = Path("data/MAGFiLO_1.0_Kaggle_2026")
TEST_DIR = D / "test" / "test_images"
H, W = 2048, 2048

DETECTOR_SCORE_THRESHOLD = 0.80
DETECTOR_MASK_THRESHOLD = 0.5
UNET_THRESHOLD = 0.8
UNET_MIN_AREA = 300
DEDUP_CONTAINMENT = 0.5  # skip a U-Net box if this much of it is already covered by a detector box
UNET_PSEUDO_SCORE = 0.5  # lower than typical detector scores so detector wins disputed pixels
MIN_AREA = 20
CHUNK_SIZE = 20
COOLDOWN_SECONDS = 15


def load_models(detector_ckpt, refiner_ckpt, unet_ckpt, device):
    detector = build_detector(num_classes=2).to(device)
    detector.load_state_dict(torch.load(detector_ckpt, map_location=device)["model"])
    detector.eval()

    refiner = RefinerUNet().to(device)
    refiner.load_state_dict(torch.load(refiner_ckpt, map_location=device)["model"])
    refiner.eval()

    unet = build_unet().to(device)
    unet.load_state_dict(torch.load(unet_ckpt, map_location=device)["model"])
    unet.eval()

    return detector, refiner, unet


def box_iou(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def contained_fraction(inner, outer):
    """Fraction of `inner` box's area that overlaps `outer` -- catches the
    case where `inner` is just a fragment of an already-detected filament,
    which plain IoU misses since a small piece has low IoU with a big box."""
    x0 = max(inner[0], outer[0]); y0 = max(inner[1], outer[1])
    x1 = min(inner[2], outer[2]); y1 = min(inner[3], outer[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    inner_area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return inter / inner_area if inner_area > 0 else 0.0


@torch.no_grad()
def detector_boxes(detector, device, img_t):
    out = detector([img_t.to(device)])[0]
    scores = out["scores"].cpu().numpy()
    masks = out["masks"].cpu().numpy()
    boxes = []
    for j in range(len(scores)):
        if scores[j] < DETECTOR_SCORE_THRESHOLD:
            continue
        m = masks[j, 0] > DETECTOR_MASK_THRESHOLD
        if m.sum() == 0:
            continue
        ys, xs = np.where(m)
        boxes.append((float(scores[j]), [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]))
    return boxes


@torch.no_grad()
def unet_boxes(unet, device, gray_img):
    out = val_transform(image=gray_img, mask=np.zeros_like(gray_img, dtype=np.float32))
    img_t = out["image"].unsqueeze(0).to(device)
    logits = unet(img_t)
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    probs_full = np.array(Image.fromarray((probs * 255).astype(np.uint8)).resize(
        (W, H), Image.BILINEAR)).astype(np.float32) / 255.0

    binary = (probs_full > UNET_THRESHOLD).astype(np.uint8)
    labeled, n = ndimage.label(binary, structure=np.ones((3, 3)))
    boxes = []
    for i in range(1, n + 1):
        m = labeled == i
        area = int(m.sum())
        if area < UNET_MIN_AREA:
            continue
        ys, xs = np.where(m)
        boxes.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])
    return boxes


@torch.no_grad()
def predict_one(detector, refiner, unet, device, gray_img, img_t):
    det_boxes = detector_boxes(detector, device, img_t)
    u_boxes = unet_boxes(unet, device, gray_img)

    combined = list(det_boxes)  # (score, box)
    n_added = 0
    for ub in u_boxes:
        if any(contained_fraction(ub, db) > DEDUP_CONTAINMENT for _, db in det_boxes):
            continue
        combined.append((UNET_PSEUDO_SCORE, ub))
        n_added += 1

    candidates = []
    for score, (x0, y0, x1, y1) in combined:
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        coarse = np.zeros((H, W), dtype=np.uint8)
        coarse[y0:y1, x0:x1] = 1
        cx0, cy0, cx1, cy1 = square_bounds(coarse)
        crop = np.array(Image.fromarray(gray_img[cy0:cy1, cx0:cx1]).resize(
            (CROP_SIZE, CROP_SIZE), Image.BILINEAR))
        prob = refine_with_tta(refiner, device, crop)

        side = cy1 - cy0
        prob_full = np.array(Image.fromarray((prob * 255).astype(np.uint8)).resize(
            (side, side), Image.BILINEAR)).astype(np.float32) / 255.0
        refined_crop = (prob_full > 0.5).astype(np.uint8)

        full_mask = np.zeros((H, W), dtype=np.uint8)
        full_mask[cy0:cy1, cx0:cx1] = refined_crop
        if full_mask.sum() == 0:
            continue
        candidates.append((score, full_mask))

    kept = paint_panoptic(candidates, MIN_AREA)
    return kept, n_added


def predict(detector_ckpt, refiner_ckpt, unet_ckpt, out_csv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector, refiner, unet = load_models(detector_ckpt, refiner_ckpt, unet_ckpt, device)
    print(f"loaded detector/refiner/unet on {device}")

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
            rgb = np.array(Image.open(path).convert("RGB"))
            img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            kept, n_added = predict_one(detector, refiner, unet, device, gray, img_t)
            stem = path.stem
            rows.extend({"filament_id": f"{stem}_{k}", "segmentation_rle": to_rle(m)}
                        for k, m in enumerate(kept, 1))
            print(f"{stem}: {len(kept)} kept ({n_added} added by U-Net)")

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


def eval_val(detector_ckpt, refiner_ckpt, unet_ckpt, n=30):
    from dataset import train_val_split, IMG_DIR
    from pq import compute_pq

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector, refiner, unet = load_models(detector_ckpt, refiner_ckpt, unet_ckpt, device)
    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)

    pqs = []
    total_added = 0
    for i, e in enumerate(val_entries[:n], 1):
        gray = np.array(Image.open(IMG_DIR / e["file_name"]).convert("L"))
        rgb = np.array(Image.open(IMG_DIR / e["file_name"]).convert("RGB"))
        img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        kept, n_added = predict_one(detector, refiner, unet, device, gray, img_t)
        total_added += n_added
        pred_rles = [to_rle(m) for m in kept]

        gt_rles = []
        for a in per_image.get(e["id"], []):
            rles = mu.frPyObjects(a["segmentation"], H, W)
            gt_rles.append(to_rle(mu.decode(mu.merge(rles))))

        pq = compute_pq(pred_rles, gt_rles, H, W)
        pqs.append(pq)
        print(f"[{i}/{n}] {e['file_name']}: pred={len(kept):3d} gt={len(gt_rles):3d} "
              f"added={n_added} PQ={pq:.3f}")
        if i % CHUNK_SIZE == 0 and i < n:
            torch.cuda.empty_cache()
            time.sleep(COOLDOWN_SECONDS)

    print(f"\nmean PQ over {len(pqs)} val images: {np.mean(pqs):.3f} "
          f"(total U-Net-added boxes: {total_added})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--detector", default="checkpoints/maskrcnn_epoch3.pt")
    p.add_argument("--refiner", default="checkpoints/refiner_best.pt")
    p.add_argument("--unet", default="checkpoints/unet_best.pt")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--eval-val", action="store_true")
    args = p.parse_args()

    if args.eval_val:
        eval_val(args.detector, args.refiner, args.unet)
    else:
        predict(args.detector, args.refiner, args.unet, args.out)
