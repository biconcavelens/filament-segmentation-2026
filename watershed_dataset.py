"""Whole-image cache + dataset for a dense distance-transform / watershed
instance segmentation model -- a genuinely different architecture from the
box-proposal (Mask R-CNN) family used everywhere else this session.

Rationale: every experiment so far has been built on anchor-based box
proposals, and the dominant, stubborn error class all session has been
total-miss (the detector never proposes a box at all for ~25-30% of GT
filaments). A dense per-pixel model has no proposal step to fail at --
every pixel gets a chance to contribute. Touching/nearby filaments, which
NMS/box-IoU struggles to cleanly separate, become two distinct local maxima
in a per-pixel distance-to-instance-boundary map, with a valley between them
at the point of contact -- watershed splits them there directly, with no
box, no anchor, no IoU threshold anywhere in the loop.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
import pycocotools.mask as mu

from dataset import IMG_DIR, H, W

PROC_SIZE = 1024   # whole-image cache resolution (vs the detector's ~800px effective resolution)
DIST_SCALE = 10.0  # raw pixel distance-to-boundary that maps to the top of the uint8 range


def _instance_distance_map(anns, h, w):
    """Per-pixel distance to the boundary of THIS pixel's own instance
    (each instance's distance transform computed independently, then
    composited -- instances are non-overlapping by construction so there's
    no ambiguity about which instance a foreground pixel belongs to)."""
    dist = np.zeros((h, w), dtype=np.float32)
    semantic = np.zeros((h, w), dtype=np.uint8)
    for a in anns:
        rles = mu.frPyObjects(a["segmentation"], h, w)
        m = mu.decode(mu.merge(rles)).astype(bool)
        if not m.any():
            continue
        d = ndimage.distance_transform_edt(m)
        dist[m] = d[m]
        semantic[m] = 1
    return semantic, dist


def build_watershed_cache(entries: list[dict], per_image: dict, cache_dir: Path, prefix: str):
    cache_dir.mkdir(parents=True, exist_ok=True)
    img_path = cache_dir / f"{prefix}_images.npy"
    if img_path.exists():
        print(f"{prefix} watershed cache already exists, skipping")
        return

    n = len(entries)
    images = np.lib.format.open_memmap(img_path, mode="w+", dtype=np.uint8, shape=(n, PROC_SIZE, PROC_SIZE))
    semantics = np.lib.format.open_memmap(cache_dir / f"{prefix}_semantic.npy", mode="w+",
                                           dtype=np.uint8, shape=(n, PROC_SIZE, PROC_SIZE))
    distances = np.lib.format.open_memmap(cache_dir / f"{prefix}_distance.npy", mode="w+",
                                           dtype=np.uint8, shape=(n, PROC_SIZE, PROC_SIZE))

    for i, e in enumerate(entries):
        gray = np.array(Image.open(IMG_DIR / e["file_name"]).convert("L"))
        anns = per_image.get(e["id"], [])
        semantic, dist = _instance_distance_map(anns, H, W)
        dist_u8 = np.clip(dist / DIST_SCALE, 0, 1) * 255

        images[i] = np.array(Image.fromarray(gray).resize((PROC_SIZE, PROC_SIZE), Image.BILINEAR))
        semantics[i] = np.array(Image.fromarray(semantic * 255).resize(
            (PROC_SIZE, PROC_SIZE), Image.NEAREST)) > 127
        distances[i] = np.array(Image.fromarray(dist_u8.astype(np.uint8)).resize(
            (PROC_SIZE, PROC_SIZE), Image.BILINEAR))

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n}", flush=True)

    images.flush(); semantics.flush(); distances.flush()
    print(f"{prefix}: {n} whole-image watershed samples cached to {cache_dir}")


class WatershedDataset(torch.utils.data.Dataset):
    """Random CROP-sized crops from the cached PROC_SIZE whole images, so a
    single cache serves many effective training samples per image."""

    def __init__(self, cache_dir: Path, prefix: str, crop: int = 384,
                 crops_per_image: int = 4, augment: bool = False):
        self.images = np.load(cache_dir / f"{prefix}_images.npy", mmap_mode="r")
        self.semantics = np.load(cache_dir / f"{prefix}_semantic.npy", mmap_mode="r")
        self.distances = np.load(cache_dir / f"{prefix}_distance.npy", mmap_mode="r")
        self.crop = crop
        self.crops_per_image = crops_per_image
        self.augment = augment

    def __len__(self):
        return len(self.images) * self.crops_per_image

    def __getitem__(self, idx):
        import random
        img_idx = idx % len(self.images)
        img = np.asarray(self.images[img_idx])
        sem = np.asarray(self.semantics[img_idx])
        dist = np.asarray(self.distances[img_idx])

        c = self.crop
        y0 = random.randint(0, PROC_SIZE - c)
        x0 = random.randint(0, PROC_SIZE - c)
        img_c = img[y0:y0 + c, x0:x0 + c].copy()
        sem_c = sem[y0:y0 + c, x0:x0 + c].copy()
        dist_c = dist[y0:y0 + c, x0:x0 + c].copy()

        if self.augment:
            k = random.randrange(4)
            img_c = np.rot90(img_c, k).copy()
            sem_c = np.rot90(sem_c, k).copy()
            dist_c = np.rot90(dist_c, k).copy()
            if random.random() < 0.5:
                img_c = np.fliplr(img_c).copy()
                sem_c = np.fliplr(sem_c).copy()
                dist_c = np.fliplr(dist_c).copy()

        x = torch.from_numpy(img_c).float().unsqueeze(0) / 255.0
        y_sem = torch.from_numpy(sem_c.astype(np.float32)).unsqueeze(0)
        y_dist = torch.from_numpy(dist_c.astype(np.float32)).unsqueeze(0) / 255.0
        return x, torch.cat([y_sem, y_dist], dim=0)


def selftest():
    import gc
    from dataset import train_val_split

    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)
    cache_dir = Path("scratch_watershed_selftest")
    build_watershed_cache(val_entries[:3], per_image, cache_dir, "selftest")

    ds = WatershedDataset(cache_dir, "selftest", crop=256, crops_per_image=2, augment=True)
    assert len(ds) == 6
    x, y = ds[0]
    assert x.shape == (1, 256, 256)
    assert y.shape == (2, 256, 256)
    assert y[0].max() <= 1.0 and y[0].min() >= 0.0
    assert y[1].max() <= 1.0 and y[1].min() >= 0.0
    print("watershed dataset selftest OK, sample semantic sum:", y[0].sum().item(),
          "distance max:", y[1].max().item())

    del ds
    gc.collect()
    for f in cache_dir.iterdir():
        f.unlink()
    cache_dir.rmdir()


if __name__ == "__main__":
    selftest()
