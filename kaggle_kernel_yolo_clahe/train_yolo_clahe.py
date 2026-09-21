"""Train YOLO11m-seg (cls=1.5, the validated confidence-calibration fix)
on CLAHE-preprocessed images instead of the raw competition JPEGs.

Motivation: the established ceiling diagnosis found the pipeline's
remaining misses are dominated by filaments at ~1/10th the local contrast
of the ones we catch (median 0.8-1.6 vs ~16.5 gray levels, out of 255) --
essentially invisible in the raw 8-bit JPEG. Investigated recovering that
contrast from the original 14-bit FITS files (see diag_fits_flatfield.py)
but 8 different flat-fielding attempts all failed to reproduce NSO's own
enhancement pipeline. CLAHE (Contrast Limited Adaptive Histogram
Equalization) is a much simpler, well-established alternative that works
directly on the existing 8-bit JPEGs: tested on a real training image's
12 GT filaments, it amplified mean local contrast from 23.0 to 50-112
(2-5x) while keeping 100% of filaments showing positive contrast, at
every clip-limit/tile-size setting tried -- the strongest, most
consistent positive signal of any preprocessing tried this session.

clipLimit=4.0, tileGridSize=8x8 chosen as a moderate middle setting
(higher clip limits amplify contrast further but risk amplifying JPEG
compression artifacts/noise along with real signal -- untested how that
trades off against detection quality, hence starting moderate rather than
at the strongest tested value).

Same training recipe as the validated cls=1.5 run (epochs=40, batch=4,
imgsz=1280) so CLAHE preprocessing is the only new variable. IMPORTANT:
inference must apply the identical CLAHE transform to test images before
running the detector, or train/inference distributions won't match.
"""
import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "opencv-python-headless"], check=True)

import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

DATA = Path("/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
IMG_DIR = DATA / "train" / "train_images"
ANN_PATH = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
YOLO_ROOT = Path("/kaggle/working/yolo_data")

H, W = 2048, 2048
CLS_GAIN = 1.5
CLAHE_CLIP = 4.0
CLAHE_TILE = 8


def apply_clahe(img_path, out_path):
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(CLAHE_TILE, CLAHE_TILE))
    enhanced = clahe.apply(gray)
    # 3-channel so YOLO's RGB-input path works unchanged
    rgb = np.stack([enhanced] * 3, axis=-1)
    Image.fromarray(rgb).save(out_path, quality=95)


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


def build_split(entries, per_image, split):
    img_dir = YOLO_ROOT / "images" / split
    lbl_dir = YOLO_ROOT / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for e in entries:
        anns = per_image.get(e["id"], [])
        name = _safe_name(e["id"])
        src_img = IMG_DIR / e["file_name"]
        dst_img = img_dir / f"{name}.jpeg"
        if not dst_img.exists():
            apply_clahe(src_img, dst_img)

        lines = []
        for a in anns:
            for poly in a["segmentation"]:
                pts = poly
                if len(pts) < 6:
                    continue
                norm = []
                for i in range(0, len(pts), 2):
                    x, y = pts[i] / W, pts[i + 1] / H
                    norm.append(f"{x:.6f} {y:.6f}")
                lines.append("0 " + " ".join(norm))
        (lbl_dir / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_written += 1
        if n_written % 100 == 0:
            print(f"  CLAHE-processed {n_written}/{len(entries)} ({split})", flush=True)
    return n_written


def build_dataset():
    train_entries, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)
    n_train = build_split(train_entries, per_image, "train")
    n_val = build_split(val_entries, per_image, "val")
    print(f"train: {n_train} images, val: {n_val} images", flush=True)

    yaml_content = f"""path: {YOLO_ROOT.resolve()}
train: images/train
val: images/val
names:
  0: filament
"""
    (YOLO_ROOT / "data.yaml").write_text(yaml_content, encoding="utf-8")
    return YOLO_ROOT / "data.yaml"


def train_with_resume(data_yaml, epochs=40, max_retries=6):
    last_ckpt = Path("/kaggle/working/runs/segment/filament_clahe/weights/last.pt")
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
                    patience=15, seed=0, deterministic=True, cls=CLS_GAIN,
                    project="/kaggle/working/runs/segment", name="filament_clahe",
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
    data_yaml = build_dataset()
    train_with_resume(data_yaml)

    best = Path("/kaggle/working/runs/segment/filament_clahe/weights/best.pt")
    out = Path("/kaggle/working/yolo11m_clahe_best.pt")
    shutil.copy(best, out)
    print(f"copied {best} -> {out}", flush=True)


if __name__ == "__main__":
    main()
