"""Classical CV baseline for filament segmentation: no training needed.

Filaments are dark, thread-like blobs on a bright solar disk. Pipeline:
  1. find the solar disk (largest bright blob, eroded inward)
  2. estimate a slowly-varying local background via heavy gaussian blur
  3. threshold pixels that are darker than their local background
  4. clean up with morphological opening, label connected components
  5. drop components outside the observed filament area range

Usage:
    python baseline.py --calibrate   # tune threshold against train GT, prints Dice
    python baseline.py --predict     # write submission.csv from test images
    python baseline.py --selftest    # synthetic sanity check
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
import pycocotools.mask as mu

D = Path("data/MAGFiLO_1.0_Kaggle_2026")
ANN_PATH = D / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
H, W = 2048, 2048

# ponytail: thresholds picked from a handful of training images (see
# --calibrate); a learned model would replace this whole heuristic stack.
DISK_BRIGHTNESS_MIN = 15
DISK_ERODE_PX = 15
BG_BLUR_SIGMA = 50
DARK_RESIDUAL_THRESHOLD = 20
OPEN_STRUCTURE_PX = 1
MIN_AREA = 150
MAX_AREA = 45000


def disk_mask(img: np.ndarray) -> np.ndarray:
    bright = img > DISK_BRIGHTNESS_MIN
    labeled, n = ndimage.label(bright)
    if n == 0:
        return np.zeros_like(img, dtype=bool)
    sizes = ndimage.sum(bright, labeled, range(1, n + 1))
    disk = labeled == (np.argmax(sizes) + 1)
    disk = ndimage.binary_fill_holes(disk)
    disk = ndimage.binary_erosion(disk, iterations=DISK_ERODE_PX)
    return disk


def segment_filaments(img: np.ndarray) -> list[np.ndarray]:
    """Return a list of disjoint boolean masks, one per detected filament."""
    disk = disk_mask(img)
    background = ndimage.gaussian_filter(img.astype(np.float32), BG_BLUR_SIGMA)
    residual = background - img.astype(np.float32)
    candidate = (residual > DARK_RESIDUAL_THRESHOLD) & disk
    candidate = ndimage.binary_opening(
        candidate, structure=np.ones((OPEN_STRUCTURE_PX * 2 + 1,) * 2)
    )

    labeled, n = ndimage.label(candidate, structure=np.ones((3, 3)))
    masks = []
    for i in range(1, n + 1):
        m = labeled == i
        area = int(m.sum())
        if MIN_AREA <= area <= MAX_AREA:
            masks.append(m)
    return masks


def to_rle(mask: np.ndarray) -> str:
    return mu.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("utf-8")


def predict():
    test_dir = D / "test" / "test_images"
    rows = []
    files = sorted(test_dir.iterdir())
    for i, path in enumerate(files, 1):
        img = np.array(Image.open(path).convert("L"))
        masks = segment_filaments(img)
        stem = path.stem
        for j, m in enumerate(masks, 1):
            rows.append({"filament_id": f"{stem}_{j}", "segmentation_rle": to_rle(m)})
        print(f"[{i}/{len(files)}] {stem}: {len(masks)} filaments")

    out = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    out.to_csv("submission.csv", index=False)
    print(f"\nwrote submission.csv: {len(out)} rows, {len(files)} images, "
          f"avg {len(out) / len(files):.1f} filaments/image")


def _load_gt_mask(anns: list[dict]) -> np.ndarray:
    combined = np.zeros((H, W), dtype=np.uint8)
    for a in anns:
        rles = mu.frPyObjects(a["segmentation"], H, W)
        combined |= mu.decode(mu.merge(rles))
    return combined.astype(bool)


def calibrate(n_images: int = 10):
    with open(ANN_PATH, encoding="utf-8") as f:
        coco = json.load(f)
    per_image = {}
    for a in coco["annotations"]:
        per_image.setdefault(a["image_id"], []).append(a)

    seen_files = set()
    sample = []
    for img_entry in coco["images"]:
        if img_entry["file_name"] in seen_files:
            continue
        seen_files.add(img_entry["file_name"])
        sample.append(img_entry)
        if len(sample) >= n_images:
            break

    dices = []
    for img_entry in sample:
        img = np.array(
            Image.open(D / "train" / "train_images" / img_entry["file_name"]).convert("L")
        )
        gt = _load_gt_mask(per_image[img_entry["id"]])
        masks = segment_filaments(img)
        pred = np.zeros((H, W), dtype=bool)
        for m in masks:
            pred |= m
        inter = (pred & gt).sum()
        dice = 2 * inter / (pred.sum() + gt.sum() + 1e-9)
        dices.append(dice)
        print(f"{img_entry['file_name']}: pred_blobs={len(masks):3d} "
              f"gt_filaments={len(per_image[img_entry['id']]):3d} pixel_dice={dice:.3f}")

    print(f"\nmean pixel dice over {len(dices)} images: {np.mean(dices):.3f}")


def selftest():
    img = np.full((300, 300), 200, dtype=np.uint8)  # bright disk
    img[20:280, 20:280] = 220
    img[100:110, 100:160] = 40  # dark thread -> should be detected
    img[250:252, 250:252] = 30  # tiny speck -> should be filtered by area

    global H, W
    H, W = img.shape
    masks = segment_filaments(img)
    assert len(masks) == 1, f"expected 1 filament, got {len(masks)}"
    detected = masks[0]
    assert detected[105, 130], "expected the dark thread region to be detected"
    assert not detected[251, 251], "tiny speck should have been filtered out"

    rle = to_rle(detected)
    roundtrip = mu.decode({"size": [H, W], "counts": rle.encode("utf-8")})
    assert np.array_equal(roundtrip.astype(bool), detected), "RLE round-trip mismatch"

    H, W = 2048, 2048
    print("selftest OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--predict", action="store_true")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    if args.selftest:
        selftest()
    elif args.calibrate:
        calibrate()
    elif args.predict:
        predict()
    else:
        print(__doc__)
        sys.exit(1)
