"""Test YOLO's own built-in test-time augmentation (Ultralytics' `augment=
True`: flip + multi-scale inference, merged internally via NMS) on top of
the confirmed-best YOLO11m config (imgsz=1280, conf=0.35, local val PQ
0.4171). Unlike the cross-architecture ensemble -- which failed three
different ways because pooling raw candidates from two separately-trained
models creates redundant overlapping proposals that panoptic-paint can't
cleanly resolve -- single-model TTA merges augmented views through
Ultralytics' own NMS *before* candidates ever reach our pipeline, so it
doesn't have that failure mode. Single-axis: augment on vs off, everything
else fixed at the validated config.
"""
import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pycocotools.mask as mu
from PIL import Image
from ultralytics import YOLO

DATA = Path("/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
IMG_DIR = DATA / "train" / "train_images"
ANN_PATH = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
CKPT = Path("/kaggle/input/datasets/trishanthmellimi/filament-seg-checkpoints")

H, W = 2048, 2048
CROP_SIZE = 256
YOLO_CONF = 0.35
MIN_AREA = 20
CONTEXT = 1.8
MIN_CROP = 96
REFINER_THRESHOLD = 0.5
TTA_AREA_RATIO_MIN = 0.60
TTA_AREA_RATIO_MAX = 1.70


class ConvBlock(nn.Sequential):
    def __init__(self, in_c, out_c):
        super().__init__(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )


class RefinerUNet(nn.Module):
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


def square_bounds(mask, context=CONTEXT, minimum=MIN_CROP):
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


@torch.no_grad()
def refine_with_tta(refiner, device, x):
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


def train_val_split(val_frac=0.1, seed=0):
    with open(ANN_PATH, encoding="utf-8") as f:
        coco = json.load(f)
    per_image = {}
    for a in coco["annotations"]:
        per_image.setdefault(a["image_id"], []).append(a)
    images = coco["images"]
    files = sorted(set(i["file_name"] for i in images))
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * val_frac))
    val_files = set(files[:n_val])
    val_entries = [i for i in images if i["file_name"] in val_files]
    return val_entries, per_image


@torch.no_grad()
def predict_one(yolo, refiner, device, gray, img_path, imgsz, augment):
    out = yolo.predict(source=str(img_path), imgsz=imgsz, conf=YOLO_CONF,
                        augment=augment, verbose=False)[0]
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


def eval_pq(yolo, refiner, device, val_entries, per_image, imgsz, augment):
    pq_num = pq_den = 0.0
    tp = 0
    for i, e in enumerate(val_entries, 1):
        img_path = IMG_DIR / e["file_name"]
        gray = np.array(Image.open(img_path).convert("L"))
        kept = predict_one(yolo, refiner, device, gray, img_path, imgsz, augment)
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
        n_fp = len(pred) - m_tp
        n_fn = len(gt) - m_tp
        pq_num += float(best_per_gt[best_per_gt > 0.5].sum())
        pq_den += m_tp + 0.5 * n_fp + 0.5 * n_fn
        if i % 20 == 0:
            print(f"  augment={augment}: {i}/{len(val_entries)}", flush=True)
    return pq_num / pq_den if pq_den else 0.0, tp


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo = YOLO(str(CKPT / "yolo11m_best.pt"))
    rstate = torch.load(CKPT / "refiner_v5_best.pt", map_location=device)
    refiner = RefinerUNet(in_channels=rstate.get("in_channels", 1),
                           out_channels=rstate.get("out_channels", 1)).to(device)
    refiner.load_state_dict(rstate["model"])
    refiner.eval()

    val_entries, per_image = train_val_split()
    print(f"val entries: {len(val_entries)}")

    for augment in (False, True):
        pq, tp = eval_pq(yolo, refiner, device, val_entries, per_image, 1280, augment)
        print(f"=== YOLO11m augment={augment}: PQ={pq:.4f} TP={tp} "
              f"(no-TTA baseline: PQ=0.4171 TP=504) ===", flush=True)


if __name__ == "__main__":
    main()
