"""Extend the winning cls=1.5 YOLO11m run (real PQ 0.37 -> 0.38) with more
training. The loss curve at epoch 40 was still improving -- val/cls_loss
still trending down (3.76 -> 3.44 over the last 14 epochs) and mAP50(M)
still climbing (0.60 -> 0.67), no sign of a plateau -- so the model likely
wasn't fully converged under the new (higher) classification-loss weight.

True ultralytics `resume=True` expects the checkpoint's own recorded
save_dir to exist, which won't be true across a fresh Kaggle kernel
filesystem -- so instead this loads the epoch-40 weights as the starting
point for a fresh continued-training run (own LR schedule/warmup), which is
more robust across machines than trying to reconstruct exact resume state.
Builds the dataset fresh from the competition's mounted COCO annotations,
same as the original run.
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
CKPT = Path("/kaggle/input/datasets/trishanthmellimi/filament-seg-checkpoints")
YOLO_ROOT = Path("/kaggle/working/yolo_data")

H, W = 2048, 2048
CLS_GAIN = 1.5
EXTRA_EPOCHS = 30


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
                if len(poly) < 6:
                    continue
                norm = []
                for i in range(0, len(poly), 2):
                    x, y = poly[i] / W, poly[i + 1] / H
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


def train_with_resume(data_yaml, epochs, max_retries=6):
    last_ckpt = Path("/kaggle/working/runs/segment/filament_cls_ext/weights/last.pt")
    attempt = 0
    while attempt <= max_retries:
        try:
            if last_ckpt.exists():
                print(f"resuming from {last_ckpt}", flush=True)
                model = YOLO(str(last_ckpt))
                model.train(resume=True)
            else:
                model = YOLO(str(CKPT / "yolo11m_cls_last.pt"))
                model.train(
                    data=str(data_yaml), epochs=epochs, batch=4, imgsz=1280,
                    patience=15, seed=0, deterministic=True, cls=CLS_GAIN,
                    project="/kaggle/working/runs/segment", name="filament_cls_ext",
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
    train_with_resume(data_yaml, epochs=EXTRA_EPOCHS)

    best = Path("/kaggle/working/runs/segment/filament_cls_ext/weights/best.pt")
    out = Path("/kaggle/working/yolo11m_cls_ext_best.pt")
    shutil.copy(best, out)
    print(f"copied {best} -> {out}", flush=True)


if __name__ == "__main__":
    main()
