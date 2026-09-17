"""Self-training: every genuinely different architecture/capacity change
tried this session (confidence tuning, imgsz, YOLO11s vs 11m, TTA) plateaus
at the same real PQ 0.37 ceiling, and three ensemble-merge strategies all
underperform solo detection. That convergence points at training-data
volume being the real bottleneck, not model choice -- so this generates
strict high-confidence pseudo-labels on the unlabeled competition test set
using the current best pipeline (YOLO11m + refiner, imgsz=1280, conf=0.35
for the base detector; pseudo-labels themselves use a much stricter 0.75
floor to keep label noise low), adds them to the real training set, and
retrains YOLO11m from scratch on the combined data.

Uses the test set's IMAGES only (for inference), never any label -- the
competition provides no test labels to leak. This is standard semi-
supervised self-training, not label leakage.

Only worth keeping if it beats the real baseline on the untouched 116-image
val split (real labels, never touched by pseudo-labeling) -- eval_val_m.py
already gives the YOLO11m baseline to compare against (PQ=0.4171).
"""
import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)

import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from ultralytics import YOLO

DATA = Path("/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
IMG_DIR = DATA / "train" / "train_images"
TEST_DIR = DATA / "test" / "test_images"
ANN_PATH = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
CKPT = Path("/kaggle/input/datasets/trishanthmellimi/filament-seg-checkpoints")
YOLO_ROOT = Path("/kaggle/working/yolo_data")

H, W = 2048, 2048
CROP_SIZE = 256
CONTEXT = 1.8
MIN_CROP = 96
MIN_AREA = 20
REFINER_THRESHOLD = 0.5
TTA_AREA_RATIO_MIN = 0.60
TTA_AREA_RATIO_MAX = 1.70
BASE_YOLO_CONF = 0.35     # base detector operating point (matches validated pipeline)
PSEUDO_LABEL_CONF = 0.75  # much stricter floor for what becomes a pseudo-label


# ---------- refiner (verified-equivalent inline copy, same as other kernels) ----------

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
        "identity": x, "hflip": x[:, :, ::-1], "vflip": x[:, ::-1, :], "hvflip": x[:, ::-1, ::-1],
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


def paint_panoptic(candidates, min_area=MIN_AREA):
    candidates = sorted(candidates, key=lambda x: -x[0])
    claimed = np.zeros((H, W), dtype=np.uint8)
    kept = []
    for score, binary in candidates:
        remaining = binary & (1 - claimed)
        area = int(remaining.sum())
        if area < min_area:
            continue
        claimed |= remaining
        kept.append((score, remaining))
    return kept


@torch.no_grad()
def predict_with_scores(yolo, refiner, device, gray, img_path):
    out = yolo.predict(source=str(img_path), imgsz=1280, conf=BASE_YOLO_CONF, verbose=False)[0]
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
    return paint_panoptic(candidates)


def mask_to_yolo_polygons(mask):
    """Binary mask -> list of normalized YOLO polygon point-lists (one per
    contour, so a mask with disjoint pieces becomes multiple polygon lines)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if len(c) < 3:
            continue
        eps = 0.002 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        norm = []
        for x, y in approx:
            norm.append(f"{x / W:.6f} {y / H:.6f}")
        polys.append("0 " + " ".join(norm))
    return polys


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
    train_entries = [i for i in images if i["file_name"] not in val_files]
    val_entries = [i for i in images if i["file_name"] in val_files]
    return train_entries, val_entries, per_image


def _safe_name(entry_id) -> str:
    return str(entry_id).replace("/", "_").replace("\\", "_")


def build_real_split(entries, per_image, split):
    img_dir = YOLO_ROOT / "images" / split
    lbl_dir = YOLO_ROOT / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for e in entries:
        anns = per_image.get(e["id"], [])
        name = _safe_name(e["id"])
        src_img = IMG_DIR / e["file_name"]
        dst_img = img_dir / f"{name}.jpeg"
        if not dst_img.exists():
            shutil.copy(src_img, dst_img)
        lines = []
        for a in anns:
            for poly in a["segmentation"]:
                if len(poly) < 6:
                    continue
                norm = []
                for i in range(0, len(poly), 2):
                    x, y = poly[i] / W, poly[i + 1] / H
                    norm.append(f"{x:.6f} {y:.6f}")
                lines.append("0 " + " ".join(norm))
        (lbl_dir / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
        n += 1
    return n


def build_pseudo_labels(yolo, refiner, device):
    img_dir = YOLO_ROOT / "images" / "train"
    lbl_dir = YOLO_ROOT / "labels" / "train"
    test_files = sorted(TEST_DIR.iterdir())
    n_used = 0
    n_polys = 0
    for i, path in enumerate(test_files, 1):
        gray = np.array(Image.open(path).convert("L"))
        kept = predict_with_scores(yolo, refiner, device, gray, path)
        strict = [(s, m) for s, m in kept if s >= PSEUDO_LABEL_CONF]
        if not strict:
            continue
        lines = []
        for score, mask in strict:
            for poly_line in mask_to_yolo_polygons(mask):
                lines.append(poly_line)
                n_polys += 1
        if not lines:
            continue
        name = f"pseudo_{path.stem}"
        dst_img = img_dir / f"{name}.jpeg"
        shutil.copy(path, dst_img)
        (lbl_dir / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_used += 1
        if i % 50 == 0:
            print(f"  pseudo-labeled {i}/{len(test_files)} test images scanned, "
                  f"{n_used} used so far", flush=True)
    return n_used, n_polys


def train_with_resume(data_yaml, epochs=40, max_retries=6):
    last_ckpt = Path("/kaggle/working/runs/segment/filament_selftrain/weights/last.pt")
    attempt = 0
    while attempt <= max_retries:
        try:
            if last_ckpt.exists():
                print(f"resuming from {last_ckpt}", flush=True)
                model = YOLO(str(last_ckpt))
                model.train(resume=True)
            else:
                model = YOLO("yolo11m-seg.pt")
                model.train(
                    data=str(data_yaml), epochs=epochs, batch=4, imgsz=1280,
                    patience=15, seed=0, deterministic=True,
                    project="/kaggle/working/runs/segment", name="filament_selftrain",
                )
            print("training finished normally", flush=True)
            return
        except Exception as e:
            attempt += 1
            print(f"training crashed (attempt {attempt}/{max_retries}): {e}", flush=True)
            if attempt > max_retries or not last_ckpt.exists():
                raise
    raise RuntimeError("exceeded max retries")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    yolo = YOLO(str(CKPT / "yolo11m_best.pt"))
    rstate = torch.load(CKPT / "refiner_v5_best.pt", map_location=device)
    refiner = RefinerUNet(in_channels=rstate.get("in_channels", 1),
                           out_channels=rstate.get("out_channels", 1)).to(device)
    refiner.load_state_dict(rstate["model"])
    refiner.eval()

    train_entries, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)
    n_train = build_real_split(train_entries, per_image, "train")
    n_val = build_real_split(val_entries, per_image, "val")
    print(f"real train: {n_train}, real val (held out, untouched): {n_val}", flush=True)

    n_pseudo, n_polys = build_pseudo_labels(yolo, refiner, device)
    print(f"pseudo-labeled {n_pseudo} test images ({n_polys} polygons total, "
          f"conf>={PSEUDO_LABEL_CONF})", flush=True)

    yaml_content = f"""path: {YOLO_ROOT.resolve()}
train: images/train
val: images/val
names:
  0: filament
"""
    data_yaml = YOLO_ROOT / "data.yaml"
    data_yaml.write_text(yaml_content, encoding="utf-8")

    train_with_resume(data_yaml)

    best = Path("/kaggle/working/runs/segment/filament_selftrain/weights/best.pt")
    out = Path("/kaggle/working/yolo11m_selftrain_best.pt")
    shutil.copy(best, out)
    print(f"copied {best} -> {out}", flush=True)


if __name__ == "__main__":
    main()
