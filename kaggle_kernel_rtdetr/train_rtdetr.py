"""Train RT-DETR (transformer-based, real-time DEtection TRansformer) as a
third ensemble member -- a genuinely different architecture family from
both existing detectors: not a two-stage RPN+RoI-head CNN (Mask R-CNN) and
not an anchor-free single-stage CNN (YOLO), but a DETR-style query-based
transformer with Hungarian matching. Different architectures have shown
complementary recall in this problem before (the whole reason the ensemble
works), so a genuinely different failure mode here is worth testing.

RT-DETR in ultralytics is detection-only (no instance masks), which is
fine: our refiner already works from boxes regardless of whether the
upstream detector natively predicts masks -- Mask R-CNN's masks and YOLO's
masks both get discarded in favor of a bounding box before refinement
anyway (see square_bounds in every predict_*.py script). So RT-DETR slots
into the exact same box-in, refined-mask-out pipeline as the other two
detectors, just needs a bbox-format (not polygon) dataset.

This first run uses ultralytics' own default hyperparameters (no cls-weight
adjustment yet) -- matching how every other detector's FIRST run this
session also started from defaults before any lesson-informed follow-up.
RT-DETR's loss is a Hungarian-matched DETR-style loss (cls + L1 + GIoU
terms), not directly comparable to YOLO's simple cls= knob or Mask R-CNN's
loss_classifier reweighting, so whether the same "undertrained confidence
head" lesson even applies here is itself an open question for a follow-up.
"""
import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)

import json
import random
import shutil
from pathlib import Path

from ultralytics import RTDETR

DATA = Path("/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
IMG_DIR = DATA / "train" / "train_images"
ANN_PATH = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
YOLO_ROOT = Path("/kaggle/working/rtdetr_data")

H, W = 2048, 2048
EPOCHS = 40
IMGSZ = 1280


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
    """Bbox-format labels (not polygon) -- RT-DETR is detection-only.
    Box derived from each polygon's own extent, same bound math the
    refiner's square_bounds and every FilamentDataset already use."""
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
                xs = poly[0::2]
                ys = poly[1::2]
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)
                cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
                bw, bh = (x1 - x0) / W, (y1 - y0) / H
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
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


def train_with_resume(data_yaml, epochs=EPOCHS, max_retries=6):
    last_ckpt = Path("/kaggle/working/runs/detect/filament_rtdetr/weights/last.pt")
    attempt = 0
    while attempt <= max_retries:
        try:
            if last_ckpt.exists():
                print(f"resuming from {last_ckpt}", flush=True)
                model = RTDETR(str(last_ckpt))
                model.train(resume=True)
            else:
                model = RTDETR("rtdetr-l.pt")
                model.train(
                    data=str(data_yaml), epochs=epochs, batch=2, imgsz=IMGSZ,
                    patience=15, seed=0, deterministic=True,
                    project="/kaggle/working/runs/detect", name="filament_rtdetr",
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

    best = Path("/kaggle/working/runs/detect/filament_rtdetr/weights/best.pt")
    out = Path("/kaggle/working/rtdetr_best.pt")
    shutil.copy(best, out)
    print(f"copied {best} -> {out}", flush=True)


if __name__ == "__main__":
    main()
