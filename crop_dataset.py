"""Per-instance crop cache + dataset for the refiner U-Net.

Mask R-CNN's internal mask head predicts at a fixed 28x28 resolution then
upsamples, which loses fine barb detail. Cropping each detection's region
from the *original* full-res image and refining it at 256x256 gives the
refiner far more effective resolution on the one filament that matters.
"""
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import pycocotools.mask as mu

from dataset import train_val_split, IMG_DIR, H, W

CROP_SIZE = 256
CONTEXT = 1.8   # crop side = bbox longest side * CONTEXT, clamped to MIN_CROP
MIN_CROP = 96


def square_bounds(mask: np.ndarray, context: float = CONTEXT, minimum: int = MIN_CROP):
    """Square crop box around a mask's bounding box, padded and clamped to image bounds."""
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


def build_crop_cache(entries: list[dict], per_image: dict, cache_dir: Path, prefix: str):
    """Rasterize every filament instance's crop once, save as memmapped .npy."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    img_path = cache_dir / f"{prefix}_images.npy"
    mask_path = cache_dir / f"{prefix}_masks.npy"
    meta_path = cache_dir / f"{prefix}_meta.npy"

    total = sum(len(per_image.get(e["id"], [])) for e in entries)
    images = np.lib.format.open_memmap(img_path, mode="w+", dtype=np.uint8,
                                        shape=(total, CROP_SIZE, CROP_SIZE))
    masks = np.lib.format.open_memmap(mask_path, mode="w+", dtype=np.uint8,
                                       shape=(total, CROP_SIZE, CROP_SIZE))
    boxes = np.zeros((total, 4), dtype=np.int32)

    pos = 0
    for e in entries:
        anns = per_image.get(e["id"], [])
        if not anns:
            continue
        img = np.array(Image.open(IMG_DIR / e["file_name"]).convert("L"))
        for a in anns:
            rles = mu.frPyObjects(a["segmentation"], H, W)
            m = mu.decode(mu.merge(rles)).astype(np.uint8)
            if m.sum() == 0:
                continue
            x0, y0, x1, y1 = square_bounds(m)
            crop_img = np.array(Image.fromarray(img[y0:y1, x0:x1]).resize(
                (CROP_SIZE, CROP_SIZE), Image.BILINEAR))
            crop_mask = np.array(Image.fromarray(m[y0:y1, x0:x1] * 255).resize(
                (CROP_SIZE, CROP_SIZE), Image.NEAREST))
            images[pos] = crop_img
            masks[pos] = (crop_mask > 127).astype(np.uint8)
            boxes[pos] = [x0, y0, x1, y1]
            pos += 1

    images.flush()
    masks.flush()
    np.save(meta_path, boxes[:pos])
    # shrink files to actual pos if any were skipped (empty masks)
    if pos < total:
        images = np.asarray(images[:pos])
        masks = np.asarray(masks[:pos])
        np.save(img_path, images)
        np.save(mask_path, masks)
    return img_path, mask_path, pos


def build_spine_crop_cache(entries: list[dict], per_image: dict, cache_dir: Path, prefix: str):
    """Like build_crop_cache, but also rasterizes each filament's manually
    annotated GT spine (centerline polyline) into the same crop frame. The
    host confirmed spine data is fair to use as auxiliary training
    supervision -- unlike clDice's self-consistency skeleton (derived from
    the model's own prediction), this is a real, human-labeled centerline,
    a much stronger geometric signal for a thin/curvilinear structure."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    img_path = cache_dir / f"{prefix}_images.npy"
    mask_path = cache_dir / f"{prefix}_masks.npy"
    spine_path = cache_dir / f"{prefix}_spines.npy"

    total = sum(len(per_image.get(e["id"], [])) for e in entries)
    images = np.lib.format.open_memmap(img_path, mode="w+", dtype=np.uint8,
                                        shape=(total, CROP_SIZE, CROP_SIZE))
    masks = np.lib.format.open_memmap(mask_path, mode="w+", dtype=np.uint8,
                                       shape=(total, CROP_SIZE, CROP_SIZE))
    spines = np.lib.format.open_memmap(spine_path, mode="w+", dtype=np.uint8,
                                        shape=(total, CROP_SIZE, CROP_SIZE))

    pos = 0
    for e in entries:
        anns = per_image.get(e["id"], [])
        if not anns:
            continue
        img = np.array(Image.open(IMG_DIR / e["file_name"]).convert("L"))
        for a in anns:
            rles = mu.frPyObjects(a["segmentation"], H, W)
            m = mu.decode(mu.merge(rles)).astype(np.uint8)
            if m.sum() == 0:
                continue

            spine_full = np.zeros((H, W), dtype=np.uint8)
            pts = np.asarray(a.get("spine") or [], dtype=np.float32).reshape(-1, 2)
            if len(pts) >= 2:
                cv2.polylines(spine_full, [pts.round().astype(np.int32)],
                               isClosed=False, color=1, thickness=3)

            x0, y0, x1, y1 = square_bounds(m)
            crop_img = np.array(Image.fromarray(img[y0:y1, x0:x1]).resize(
                (CROP_SIZE, CROP_SIZE), Image.BILINEAR))
            crop_mask = np.array(Image.fromarray(m[y0:y1, x0:x1] * 255).resize(
                (CROP_SIZE, CROP_SIZE), Image.NEAREST))
            crop_spine = np.array(Image.fromarray(spine_full[y0:y1, x0:x1] * 255).resize(
                (CROP_SIZE, CROP_SIZE), Image.NEAREST))

            images[pos] = crop_img
            masks[pos] = (crop_mask > 127).astype(np.uint8)
            spines[pos] = (crop_spine > 127).astype(np.uint8)
            pos += 1

    images.flush(); masks.flush(); spines.flush()
    if pos < total:
        np.save(img_path, np.asarray(images[:pos]))
        np.save(mask_path, np.asarray(masks[:pos]))
        np.save(spine_path, np.asarray(spines[:pos]))
    return img_path, mask_path, spine_path, pos


class SpineCropDataset(torch.utils.data.Dataset):
    """Returns (image, mask, spine) triples: mask is the main target, spine
    is an auxiliary centerline target trained jointly, ignored at inference."""

    def __init__(self, img_path: Path, mask_path: Path, spine_path: Path, augment: bool = False):
        self.images = np.load(img_path, mmap_mode="r")
        self.masks = np.load(mask_path, mmap_mode="r")
        self.spines = np.load(spine_path, mmap_mode="r")
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.from_numpy(np.asarray(self.images[idx]).copy()).float() / 255.0
        mask = torch.from_numpy(np.asarray(self.masks[idx]).copy()).float()
        spine = torch.from_numpy(np.asarray(self.spines[idx]).copy()).float()
        img, mask, spine = img.unsqueeze(0), mask.unsqueeze(0), spine.unsqueeze(0)

        if self.augment:
            k = torch.randint(0, 4, (1,)).item()
            img = torch.rot90(img, k, (1, 2))
            mask = torch.rot90(mask, k, (1, 2))
            spine = torch.rot90(spine, k, (1, 2))
            if torch.rand(1).item() < 0.5:
                img, mask, spine = torch.flip(img, (2,)), torch.flip(mask, (2,)), torch.flip(spine, (2,))
            if torch.rand(1).item() < 0.5:
                img = torch.clamp(img * float(torch.empty(1).uniform_(0.85, 1.15)), 0, 1)

        target = torch.cat([mask, spine], dim=0)  # (2, H, W): channel 0=mask, 1=spine
        return img, target


def degrade_mask(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    """Make a clean GT crop mask look like a detector's coarse output:
    blurry (28x28-ish upsampled), slightly dilated/eroded, sometimes cut
    short, slightly offset. This is what the refiner will see as its hint
    channel at inference, so train on the same kind of imperfection."""
    from scipy import ndimage

    s = rng.choice([28, 40, 56])
    small = Image.fromarray(mask.astype(np.uint8) * 255).resize((s, s), Image.BILINEAR)
    m = (np.array(small.resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR)) > 127).astype(np.uint8)

    it = rng.randint(0, 2)
    if it:
        op = ndimage.binary_dilation if rng.random() < 0.5 else ndimage.binary_erosion
        m = op(m, iterations=it).astype(np.uint8)

    if rng.random() < 0.3:  # detector cut the filament short
        xs = np.where(m.any(axis=0))[0]
        if len(xs) > 10:
            cut = int(len(xs) * rng.uniform(0.15, 0.4))
            if rng.random() < 0.5:
                m[:, xs[-cut]:] = 0
            else:
                m[:, :xs[cut]] = 0

    m = np.roll(m, (rng.randint(-6, 6), rng.randint(-6, 6)), axis=(0, 1))
    return m


class HintCropDataset(torch.utils.data.Dataset):
    """2-channel input: [gray crop, degraded GT mask]. The hint tells the
    refiner WHICH filament to segment when several share the crop -- the
    1-channel refiner has to guess, which is where near-misses come from."""

    def __init__(self, img_path, mask_path, augment: bool = False):
        self.images = np.load(img_path, mmap_mode="r")
        self.masks = np.load(mask_path, mmap_mode="r")
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = np.asarray(self.images[idx]).copy()
        mask = np.asarray(self.masks[idx]).copy()
        rng = random if self.augment else random.Random(idx)  # val hints reproducible
        hint = degrade_mask(mask, rng)

        x = torch.stack([torch.from_numpy(img).float() / 255.0,
                         torch.from_numpy(hint).float()], 0)
        y = torch.from_numpy(mask).float().unsqueeze(0)

        if self.augment:
            k = random.randrange(4)
            x, y = torch.rot90(x, k, (1, 2)), torch.rot90(y, k, (1, 2))
            if random.random() < 0.5:
                x, y = torch.flip(x, (2,)), torch.flip(y, (2,))
            if random.random() < 0.5:
                x[0] = torch.clamp(x[0] * random.uniform(0.85, 1.15), 0, 1)
        return x, y


class CropDataset(torch.utils.data.Dataset):
    def __init__(self, img_path: Path, mask_path: Path, augment: bool = False):
        self.images = np.load(img_path, mmap_mode="r")
        self.masks = np.load(mask_path, mmap_mode="r")
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.from_numpy(np.asarray(self.images[idx]).copy()).float() / 255.0
        mask = torch.from_numpy(np.asarray(self.masks[idx]).copy()).float()
        img, mask = img.unsqueeze(0), mask.unsqueeze(0)

        if self.augment:
            k = torch.randint(0, 4, (1,)).item()
            img, mask = torch.rot90(img, k, (1, 2)), torch.rot90(mask, k, (1, 2))
            if torch.rand(1).item() < 0.5:
                img, mask = torch.flip(img, (2,)), torch.flip(mask, (2,))
            if torch.rand(1).item() < 0.5:
                img = torch.clamp(img * float(torch.empty(1).uniform_(0.85, 1.15)), 0, 1)

        return img, mask


def selftest():
    import gc

    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)
    cache_dir = Path("scratch_crop_cache_selftest")
    img_path, mask_path, n = build_crop_cache(val_entries[:3], per_image, cache_dir, "selftest")
    assert n > 0, "expected at least one crop"

    ds = CropDataset(img_path, mask_path, augment=True)
    img, mask = ds[0]
    assert img.shape == (1, CROP_SIZE, CROP_SIZE)
    assert mask.shape == (1, CROP_SIZE, CROP_SIZE)
    assert set(mask.unique().tolist()) <= {0.0, 1.0}
    assert mask.sum() > 0, "expected non-empty mask in a real filament crop"

    del ds, img, mask
    gc.collect()  # release numpy memmap file handles (Windows locks open files)
    for f in cache_dir.iterdir():
        f.unlink()
    cache_dir.rmdir()
    print(f"selftest OK ({n} crops from 3 val entries)")


if __name__ == "__main__":
    selftest()
