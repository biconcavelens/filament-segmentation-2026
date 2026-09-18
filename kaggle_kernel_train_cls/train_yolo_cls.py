"""Train YOLO11m-seg with a higher classification-loss weight (cls=1.5,
up from ultralytics' default 0.5), targeting a root cause found by
diag_missed_filaments.py: several large, visually obvious missed filaments
have raw YOLO boxes with GOOD localization (bbox-IoU 0.70-0.87 vs GT) but
catastrophically low confidence (as low as 0.05, vs our 0.35 deployment
threshold) -- feeding that exact box to the refiner gave IoU=0.75 vs GT.
Three different post-hoc rescue strategies (refiner confidence, contrast
filtering) failed to recover this cheaply, so this targets it upstream:
weighting the classification/confidence loss more heavily during training
so the model's own confidence better reflects box quality. Everything else
matches the validated recipe (epochs=40, batch=4, imgsz=1280, YOLO11m) so
cls weight is the only variable.

Builds the YOLO-format dataset directly from the competition's mounted COCO
annotations (no need to upload our own copy of the images). Wraps training
in a retry loop that resumes from last.pt on failure, mirroring the manual
crash-recovery this session needed locally (cuDNN/OOM errors mid-run).
"""
import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)

import json
import random
import shutil
from pathlib import Path

from ultralytics import YOLO

DATA = Path("/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
IMG_DIR = DATA / "train" / "train_images"
ANN_PATH = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
YOLO_ROOT = Path("/kaggle/working/yolo_data")

H, W = 2048, 2048


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
            shutil.copy(src_img, dst_img)

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


CLS_GAIN = 1.5  # up from ultralytics' default 0.5 -- the lever under test


def train_with_resume(data_yaml, epochs=40, max_retries=6):
    last_ckpt = Path("/kaggle/working/runs/segment/filament_cls/weights/last.pt")
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
                    project="/kaggle/working/runs/segment", name="filament_cls",
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

    best = Path("/kaggle/working/runs/segment/filament_cls/weights/best.pt")
    out = Path("/kaggle/working/yolo11m_cls_best.pt")
    shutil.copy(best, out)
    print(f"copied {best} -> {out}", flush=True)


if __name__ == "__main__":
    main()
