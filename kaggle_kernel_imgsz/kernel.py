"""Self-contained Kaggle kernel: YOLO11s-seg + crop-refine U-Net, at the
confirmed-best real config (imgsz=1280, conf=0.35, real PQ 0.37).

Reverted here after two resolution experiments: imgsz=1536 alone tied the
real score without a genuine gain, and imgsz=1792 with confidence re-tuned
(found via a 2-axis grid sweep, local PQ 0.4187 -- the largest local gain of
the whole YOLO line) actually *regressed* the real score to 0.36 despite
looking like the best local result yet. See README.md's "What's been tried
and rejected" section: a smooth-looking 2-axis sweep over a 116-image val
set still overfits, the same failure mode as the earlier ensemble grid
search. Do not re-raise IMGSZ/YOLO_CONF together without new evidence.

Runs entirely on Kaggle's GPU so it doesn't contend with local GPU use.
Ultralytics's own square-letterbox resize handles the upscale; everything
else (refiner, panoptic paint, RLE) is copied inline since we can't import
the local repo's modules here.
"""
import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import pycocotools.mask as mu

DATA = Path("/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
TEST_DIR = DATA / "test" / "test_images"
CKPT = Path("/kaggle/input/datasets/trishanthmellimi/filament-seg-checkpoints")

H, W = 2048, 2048  # matches dataset.py
CROP_SIZE = 256
YOLO_CONF = 0.35  # confirmed-best real config -- see module docstring
MIN_AREA = 20
IMGSZ = 1280  # confirmed-best real config -- see module docstring


class ConvBlock(nn.Sequential):
    def __init__(self, in_c, out_c):
        super().__init__(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )


class RefinerUNet(nn.Module):
    """Exact copy of train_refiner.py's architecture -- must match the
    checkpoint's state_dict exactly (bias=False convs, 'head' not 'out')."""

    def __init__(self, features=32, in_channels=1, out_channels=1):
        super().__init__()
        f = features
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = ConvBlock(f, f * 2)
        self.enc3 = ConvBlock(f * 2, f * 4)
        self.enc4 = ConvBlock(f * 4, f * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(f * 8, f * 16)
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, 2, 2)
        self.dec4 = ConvBlock(f * 16, f * 8)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, 2)
        self.dec3 = ConvBlock(f * 8, f * 4)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, 2)
        self.dec2 = ConvBlock(f * 4, f * 2)
        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, 2)
        self.dec1 = ConvBlock(f * 2, f)
        self.head = nn.Conv2d(f, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


CONTEXT = 1.8
MIN_CROP = 96


def square_bounds(mask, context=CONTEXT, minimum=MIN_CROP):
    """Exact copy of crop_dataset.py's square_bounds."""
    height, width = mask.shape
    ys, xs = np.where(mask)
    cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    side = int(np.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * context))
    side = min(max(side, minimum), min(height, width))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0:
        x1 -= x0; x0 = 0
    if y0 < 0:
        y1 -= y0; y0 = 0
    if x1 > width:
        x0 -= x1 - width; x1 = width
    if y1 > height:
        y0 -= y1 - height; y1 = height
    return x0, y0, x1, y1


REFINER_THRESHOLD = 0.5
TTA_AREA_RATIO_MIN = 0.60
TTA_AREA_RATIO_MAX = 1.70


@torch.no_grad()
def refine_with_tta(refiner, device, x: np.ndarray) -> np.ndarray:
    """Exact copy of predict_refined.py's refine_with_tta. x: (C,256,256)."""
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


def to_rle(mask):
    return mu.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("utf-8")


def paint_panoptic(candidates, min_area=20):
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


@torch.no_grad()
def predict_one(yolo, refiner, device, gray, img_path):
    out = yolo.predict(source=str(img_path), imgsz=IMGSZ, conf=YOLO_CONF, verbose=False)[0]
    if out.boxes is None or len(out.boxes) == 0:
        return []
    boxes = out.boxes.xyxy.cpu().numpy()
    scores = out.boxes.conf.cpu().numpy()

    candidates = []
    for j in range(len(boxes)):
        x0, y0, x1, y1 = boxes[j]
        x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
        if x1 <= x0 or y1 <= y0:
            continue
        coarse = np.zeros((H, W), dtype=np.uint8)
        coarse[y0:y1, x0:x1] = 1

        cx0, cy0, cx1, cy1 = square_bounds(coarse.astype(bool))
        crop = np.array(Image.fromarray(gray[cy0:cy1, cx0:cx1]).resize(
            (CROP_SIZE, CROP_SIZE), Image.BILINEAR)).astype(np.float32) / 255.0
        prob = refine_with_tta(refiner, device, np.stack([crop]))
        side = cy1 - cy0
        prob_full = np.array(Image.fromarray((prob * 255).astype(np.uint8)).resize(
            (side, side), Image.BILINEAR)).astype(np.float32) / 255.0
        refined_crop = (prob_full > 0.5).astype(np.uint8)
        full_mask = np.zeros((H, W), dtype=np.uint8)
        full_mask[cy0:cy1, cx0:cx1] = refined_crop
        if full_mask.sum() == 0:
            continue
        candidates.append((float(scores[j]), full_mask))

    return paint_panoptic(candidates, MIN_AREA)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo = YOLO(str(CKPT / "yolo_best.pt"))
    rstate = torch.load(CKPT / "refiner_v5_best.pt", map_location=device)
    refiner = RefinerUNet(in_channels=rstate.get("in_channels", 1),
                           out_channels=rstate.get("out_channels", 1)).to(device)
    refiner.load_state_dict(rstate["model"])
    refiner.eval()
    print(f"loaded models on {device}, imgsz={IMGSZ}")

    files = sorted(TEST_DIR.iterdir())
    rows = []
    for i, path in enumerate(files, 1):
        gray = np.array(Image.open(path).convert("L"))
        kept = predict_one(yolo, refiner, device, gray, path)
        stem = path.stem
        rows.extend({"filament_id": f"{stem}_{k}", "segmentation_rle": to_rle(m)}
                    for k, m in enumerate(kept, 1))
        if i % 20 == 0:
            print(f"{i}/{len(files)}", flush=True)

    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(
        "submission.csv", index=False)
    print(f"wrote submission.csv: {len(rows)} rows, {len(files)} images")


if __name__ == "__main__":
    main()
