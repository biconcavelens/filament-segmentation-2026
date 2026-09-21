"""Investigates whether the competition's 8-bit JPEG images are throwing
away recoverable filament contrast, by comparing against the original
14-bit FITS files from NSO's public GONG archive (legitimate under
Kaggle's External Data rules: publicly available, free, equally
accessible -- see the training JSON's own `url` field, e.g.
"https://gong2.nso.edu/HA/hag/201406/20140609/20140609195854Bh.jpg";
the FITS equivalent lives at the analogous /HA/haf/ path as
<timestamp>.fits.fz).

Confirmed: FITS files are pixel-aligned to the same 2048x2048 grid
(correlation 0.88 with the JPEG) with genuinely more precision (14-bit,
16384 levels vs JPEG's 256) -- but recovering usable filament contrast
from that extra precision requires reproducing NSO's own "Fourier
Transform digital filtering" enhancement step (per the MAGFiLO dataset
paper), and every attempt at approximating it here failed: divisive and
subtractive Gaussian-based flat-fielding at multiple scales, and even
truly raw unprocessed FITS, all show near-zero or negative mean filament
contrast, vs the JPEG's consistently positive contrast on every filament
in the test image. Likely explanation: real observatory pipelines
flat-field using dedicated instrument calibration frames (dark/flat
frames specific to the camera), not a synthetic flat derived by blurring
the science image itself -- which is all that's attempted here, and is a
fundamentally weaker substitute we don't have access to.

Requires: pip install astropy scipy (into whichever Python environment
actually runs this -- this session hit a python/pip mismatch between a
sandboxed venv and the system install; use the system Python if `python`
resolves to a restricted venv).

Usage: point FITS_URL at a matching FITS file (already downloaded once
via WebFetch during the investigation) and JPG_PATH at the corresponding
training image, then run.
"""
import json

import numpy as np
import pycocotools.mask as mu
from astropy.io import fits
from PIL import Image
from scipy.ndimage import gaussian_filter

FITS_PATH = "scratch_test.fits.fz"  # download once, e.g. via WebFetch, before running
JPG_PATH = "data/MAGFiLO_1.0_Kaggle_2026/train/train_images/20140609195854Bh.jpeg"
ANN_PATH = "data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMAGE_ID = "040301-20140609195854Bh"
H, W = 2048, 2048


def contrast_annulus(img, mask, pad=8):
    """Mean intensity of a proper annulus around the filament (padded
    bbox minus the filament mask itself) minus the filament's own mean --
    positive means the filament is genuinely darker than its surroundings."""
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    outer = np.zeros_like(mask, dtype=bool)
    outer[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = True
    annulus = outer & ~mask
    return float(img[annulus].mean() - img[mask].mean())


def load_gt_masks():
    with open(ANN_PATH, encoding="utf-8") as f:
        d = json.load(f)
    anns = [a for a in d["annotations"] if a["image_id"] == IMAGE_ID]
    return [mu.decode(mu.merge(mu.frPyObjects(a["segmentation"], H, W))).astype(bool)
            for a in anns]


def main():
    masks = load_gt_masks()
    print(f"{len(masks)} GT filaments in {IMAGE_ID}")

    jpg = np.array(Image.open(JPG_PATH).convert("L")).astype(np.float64)
    jpg_c = [contrast_annulus(jpg, m) for m in masks]
    print(f"JPEG:            mean_contrast={np.mean(jpg_c):.2f}  positive_frac={np.mean(np.array(jpg_c) > 0):.2f}")

    hdul = fits.open(FITS_PATH)
    data = hdul[1].data.astype(np.float64)
    disk_mask = data > np.percentile(data[data > 0], 5)

    raw_c = [contrast_annulus(data, m) for m in masks]
    print(f"raw FITS:        mean_contrast={np.mean(raw_c):.2f}  positive_frac={np.mean(np.array(raw_c) > 0):.2f}")

    for sigma in [60, 300, 500, 800]:
        blurred = gaussian_filter(data, sigma=sigma)
        blurred[blurred < 1] = 1
        flat = data / blurred
        lo, hi = np.percentile(flat[disk_mask], [1, 99])
        scaled = np.clip((flat - lo) / (hi - lo) * 255, 0, 255)
        c = [contrast_annulus(scaled, m) for m in masks]
        print(f"divisive  sigma={sigma:4d}: mean_contrast={np.mean(c):6.2f}  positive_frac={np.mean(np.array(c) > 0):.2f}")

    for sigma in [30, 60, 100, 150]:
        blurred = gaussian_filter(data, sigma=sigma)
        unsharp = data - blurred
        lo, hi = np.percentile(unsharp[disk_mask], [1, 99])
        scaled = np.clip((unsharp - lo) / (hi - lo) * 255, 0, 255)
        c = [contrast_annulus(scaled, m) for m in masks]
        print(f"subtract  sigma={sigma:4d}: mean_contrast={np.mean(c):6.2f}  positive_frac={np.mean(np.array(c) > 0):.2f}")


if __name__ == "__main__":
    main()
