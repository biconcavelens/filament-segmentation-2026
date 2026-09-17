"""Two-stage inference: Mask R-CNN detector -> per-instance crop -> refiner U-Net.

Each detection's box is cropped from the *original* full-res image (not the
detector's coarse internal mask), resized to 256x256, refined, then pasted
back at native resolution. Recovers fine barb detail the detector's 28x28
mask head loses.

Usage:
    python predict_refined.py --out submission.csv                # full test set
    python predict_refined.py --eval-val                          # local PQ on val split
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
import pycocotools.mask as mu

from train import build_from_checkpoint
from train_refiner import RefinerUNet
from crop_dataset import square_bounds, CROP_SIZE
from predict_trained import paint_panoptic, to_rle

D = Path("data/MAGFiLO_1.0_Kaggle_2026")
TEST_DIR = D / "test" / "test_images"
H, W = 2048, 2048

import os
DETECTOR_SCORE_THRESHOLD = float(os.environ.get("DET_THRESH", 0.80))
DETECTOR_NMS_THRESH = float(os.environ.get("DET_NMS", 0.30))
# nms=0.30 beats torchvision's box-head default (0.5) on full val PQ (0.3902
# vs 0.3864) -- swept in sweep_thresh.py, confirmed via confirm_winner.py.
DETECTOR_MASK_THRESHOLD = 0.5
REFINER_THRESHOLD = float(os.environ.get("REFINER_THRESH", 0.5))
MIN_AREA = 20
CHUNK_SIZE = 20
COOLDOWN_SECONDS = 15

# 4-view flip TTA with a topology-safe fallback, matching the recipe from
# the honest 0.70 community notebook: average identity/hflip/vflip/hvflip,
# but reject the average (fall back to identity) if it distorts the shape.
TTA_AREA_RATIO_MIN = 0.60
TTA_AREA_RATIO_MAX = 1.70


def refine_with_tta(refiner, device, x: np.ndarray) -> np.ndarray:
    """x: (C,256,256) float32 in [0,1] -- gray crop, plus the detector's coarse
    mask as a hint channel for the 2-channel refiner. Returns a 256x256
    probability map."""
    views = {
        "identity": x,
        "hflip": x[:, :, ::-1],
        "vflip": x[:, ::-1, :],
        "hvflip": x[:, ::-1, ::-1],
    }
    probs = {}
    for name, view in views.items():
        x_t = torch.from_numpy(np.ascontiguousarray(view)).float().unsqueeze(0)
        p = torch.sigmoid(refiner(x_t.to(device)))[0, 0].cpu().numpy()
        if name == "hflip":
            p = np.fliplr(p)
        elif name == "vflip":
            p = np.flipud(p)
        elif name == "hvflip":
            p = np.flipud(np.fliplr(p))
        probs[name] = p

    identity_area = (probs["identity"] > REFINER_THRESHOLD).sum()
    avg_prob = np.mean(list(probs.values()), axis=0)
    if identity_area == 0:
        return avg_prob

    avg_area = (avg_prob > REFINER_THRESHOLD).sum()
    ratio = avg_area / identity_area
    if ratio < TTA_AREA_RATIO_MIN or ratio > TTA_AREA_RATIO_MAX:
        return probs["identity"]
    return avg_prob


def load_models(detector_ckpt: str, refiner_ckpt: str, device):
    state = torch.load(detector_ckpt, map_location=device)
    detector = build_from_checkpoint(state, num_classes=2).to(device)
    detector.load_state_dict(state["model"])
    detector.roi_heads.nms_thresh = DETECTOR_NMS_THRESH
    detector.eval()

    rstate = torch.load(refiner_ckpt, map_location=device)
    refiner = RefinerUNet(in_channels=rstate.get("in_channels", 1),
                           out_channels=rstate.get("out_channels", 1)).to(device)
    refiner.load_state_dict(rstate["model"])
    refiner.eval()
    return detector, refiner


@torch.no_grad()
def predict_one(detector, refiner, device, gray_img: np.ndarray, img_t: torch.Tensor):
    """gray_img: full-res HxW uint8 array. img_t: 3xHxW float tensor (RGB) for the detector."""
    out = detector([img_t.to(device)])[0]
    scores = out["scores"].cpu().numpy()
    masks = out["masks"].cpu().numpy()

    candidates = []  # (score, refined_full_mask)
    for j in range(len(scores)):
        if scores[j] < DETECTOR_SCORE_THRESHOLD:
            continue
        coarse = (masks[j, 0] > DETECTOR_MASK_THRESHOLD).astype(np.uint8)
        if coarse.sum() == 0:
            continue

        x0, y0, x1, y1 = square_bounds(coarse)
        crop = np.array(Image.fromarray(gray_img[y0:y1, x0:x1]).resize(
            (CROP_SIZE, CROP_SIZE), Image.BILINEAR)).astype(np.float32) / 255.0
        channels = [crop]
        if refiner.in_channels == 2:
            hint = np.array(Image.fromarray(coarse[y0:y1, x0:x1] * 255).resize(
                (CROP_SIZE, CROP_SIZE), Image.NEAREST)) > 127
            channels.append(hint.astype(np.float32))
        prob = refine_with_tta(refiner, device, np.stack(channels))

        side = y1 - y0
        prob_full = np.array(Image.fromarray((prob * 255).astype(np.uint8)).resize(
            (side, side), Image.BILINEAR)).astype(np.float32) / 255.0
        refined_crop = (prob_full > REFINER_THRESHOLD).astype(np.uint8)

        full_mask = np.zeros((H, W), dtype=np.uint8)
        full_mask[y0:y1, x0:x1] = refined_crop
        if full_mask.sum() == 0:
            continue
        candidates.append((float(scores[j]), full_mask))

    kept = paint_panoptic(candidates, MIN_AREA)
    return kept


def predict(detector_ckpt: str, refiner_ckpt: str, out_csv: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector, refiner = load_models(detector_ckpt, refiner_ckpt, device)
    print(f"loaded detector={detector_ckpt} refiner={refiner_ckpt} on {device}")

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
            kept = predict_one(detector, refiner, device, gray, img_t)
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


def eval_val(detector_ckpt: str, refiner_ckpt: str, n: int = 30):
    from dataset import train_val_split, IMG_DIR
    from pq import compute_pq

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector, refiner = load_models(detector_ckpt, refiner_ckpt, device)
    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)

    pqs = []
    for i, e in enumerate(val_entries[:n], 1):
        gray = np.array(Image.open(IMG_DIR / e["file_name"]).convert("L"))
        rgb = np.array(Image.open(IMG_DIR / e["file_name"]).convert("RGB"))
        img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        kept = predict_one(detector, refiner, device, gray, img_t)
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
    p.add_argument("--detector", default="checkpoints/maskrcnn_epoch3.pt")
    p.add_argument("--refiner", default="checkpoints/refiner_best.pt")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--eval-val", action="store_true")
    args = p.parse_args()

    if args.eval_val:
        eval_val(args.detector, args.refiner)
    else:
        predict(args.detector, args.refiner, args.out)
