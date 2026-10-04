"""COCO-polygon -> torchvision detection dataset for filament instances.

Submission format only needs masks (no category), so this is single-class:
every filament gets label=1, background=0.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import pycocotools.mask as mu

D = Path("data/MAGFiLO_1.0_Kaggle_2026")
ANN_PATH = D / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMG_DIR = D / "train" / "train_images"
H, W = 2048, 2048


def _load_coco():
    with open(ANN_PATH, encoding="utf-8") as f:
        coco = json.load(f)
    per_image = {}
    for a in coco["annotations"]:
        per_image.setdefault(a["image_id"], []).append(a)
    return coco["images"], per_image


def train_val_split(val_frac=0.1, seed=0, train_labels="all"):
    """Split by underlying file_name so multi-annotator entries of the same
    image never straddle train/val (that would leak the image, not just the
    label, across the split).

    train_labels="complete" keeps only the most complete annotator entry per
    training image: 42% of images have 2-3 annotators who disagree on *which*
    filaments to mark, so training on all entries teaches the detector to
    sometimes skip real filaments. Val always keeps every entry."""
    images, per_image = _load_coco()
    files = sorted(set(i["file_name"] for i in images))
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = 0 if val_frac == 0 else max(1, int(len(files) * val_frac))
    val_files = set(files[:n_val])

    train_entries = [i for i in images if i["file_name"] not in val_files]
    val_entries = [i for i in images if i["file_name"] in val_files]

    if train_labels == "complete":
        def richness(e):
            anns = per_image.get(e["id"], [])
            return (len(anns), sum(a["area"] for a in anns))
        best = {}
        for e in train_entries:
            if e["file_name"] not in best or richness(e) > richness(best[e["file_name"]]):
                best[e["file_name"]] = e
        train_entries = list(best.values())
    return train_entries, val_entries, per_image


def _copy_paste(img_arr, raw_masks, entries, per_image, max_paste=2):
    """Paste 0-2 real filament instances from OTHER training images onto this
    one, at random locations. Directly increases the detector's exposure to
    varied instance positions/counts per image without needing new raw data
    -- targets total-miss (recall), the dominant error class per diag.py."""
    for _ in range(random.randint(0, max_paste)):
        src = random.choice(entries)
        anns = per_image.get(src["id"], [])
        if not anns:
            continue
        a = random.choice(anns)
        rles = mu.frPyObjects(a["segmentation"], H, W)
        src_mask = mu.decode(mu.merge(rles)).astype(np.uint8)
        if src_mask.sum() == 0:
            continue
        src_img = np.array(Image.open(IMG_DIR / src["file_name"]).convert("RGB"))

        ys, xs = np.where(src_mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        crop_mask = src_mask[y0:y1, x0:x1].astype(bool)
        crop_img = src_img[y0:y1, x0:x1]
        h, w = crop_mask.shape
        if h >= H or w >= W:
            continue

        py, px = random.randint(0, H - h), random.randint(0, W - w)
        paste_mask = np.zeros((H, W), dtype=np.uint8)
        paste_mask[py:py + h, px:px + w] = crop_mask

        region = img_arr[py:py + h, px:px + w]
        region[crop_mask] = crop_img[crop_mask]
        img_arr[py:py + h, px:px + w] = region

        # later paste occludes earlier masks, like a real overlapping object would
        raw_masks = [(m & ~paste_mask) for m in raw_masks]
        raw_masks.append(paste_mask)
    return img_arr, [m for m in raw_masks if m.sum() > 0]


def _random_tile_crop(img_arr, raw_masks, tile_size, prob=0.5):
    """With `prob`, train on a random tile_size x tile_size crop instead of
    the full image. Whole-image training always resizes 2048->~800px (a
    fixed 0.39 scale), so the detector never learns what a filament looks
    like at the gentler ~0.7 scale a tile gets resized to -- exactly the
    scale tiled_diag.py deploys at inference to recover sub-pixel filaments.
    Without this, the detector floods tiles with junk it was never
    calibrated for (confirmed empirically: total-miss dropped but spurious
    FPs more than tripled)."""
    if random.random() >= prob:
        return img_arr, raw_masks
    h, w = img_arr.shape[:2]
    if h <= tile_size or w <= tile_size:
        return img_arr, raw_masks
    y0 = random.randint(0, h - tile_size)
    x0 = random.randint(0, w - tile_size)
    crop_img = np.ascontiguousarray(img_arr[y0:y0 + tile_size, x0:x0 + tile_size])
    crop_masks = []
    for m in raw_masks:
        cm = m[y0:y0 + tile_size, x0:x0 + tile_size]
        if cm.sum() > 0:
            crop_masks.append(np.ascontiguousarray(cm))
    return crop_img, crop_masks


class FilamentDataset(torch.utils.data.Dataset):
    def __init__(self, entries: list[dict], per_image: dict, augment: bool = False,
                 copy_paste: bool = False, tile_aug: bool = False, tile_size: int = 1152,
                 rot90: bool = False, photometric: bool = False):
        self.entries = entries
        self.rot90 = rot90  # transpose -> with the two flips covers all 8 dihedral orientations
        self.photometric = photometric  # GONG sites differ in exposure/contrast
        self.per_image = per_image
        self.augment = augment
        self.copy_paste = copy_paste
        self.tile_aug = tile_aug
        self.tile_size = tile_size

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img_arr = np.array(Image.open(IMG_DIR / entry["file_name"]).convert("RGB"))

        anns = self.per_image.get(entry["id"], [])
        raw_masks = []
        for a in anns:
            rles = mu.frPyObjects(a["segmentation"], H, W)
            m = mu.decode(mu.merge(rles)).astype(np.uint8)
            if m.sum() == 0:
                continue
            raw_masks.append(m)

        if self.augment:
            if self.copy_paste:
                img_arr = np.ascontiguousarray(img_arr)
                img_arr, raw_masks = _copy_paste(img_arr, raw_masks, self.entries, self.per_image)
            if self.tile_aug:
                img_arr, raw_masks = _random_tile_crop(img_arr, raw_masks, self.tile_size)
            if random.random() < 0.5:
                img_arr = np.ascontiguousarray(img_arr[:, ::-1, :])
                raw_masks = [np.ascontiguousarray(m[:, ::-1]) for m in raw_masks]
            if random.random() < 0.5:
                img_arr = np.ascontiguousarray(img_arr[::-1, :, :])
                raw_masks = [np.ascontiguousarray(m[::-1, :]) for m in raw_masks]
            if self.rot90 and random.random() < 0.5:
                img_arr = np.ascontiguousarray(img_arr.transpose(1, 0, 2))
                raw_masks = [np.ascontiguousarray(m.T) for m in raw_masks]
            if self.photometric:
                x = (img_arr.astype(np.float32) / 255.0) ** random.uniform(0.8, 1.25)
                x = (x - 0.5) * random.uniform(0.8, 1.2) + 0.5 + random.uniform(-0.08, 0.08)
                disk = img_arr > 8  # leave the black off-disk background untouched
                img_arr = np.where(disk, np.clip(x, 0, 1) * 255, img_arr).astype(np.uint8)

        img = torch.from_numpy(img_arr).permute(2, 0, 1).float() / 255.0

        # boxes recomputed post-flip so they stay consistent with the mask
        masks, boxes = [], []
        for m in raw_masks:
            ys, xs = np.where(m)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            masks.append(m)

        if masks:
            masks_t = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.ones(len(masks), dtype=torch.int64)
        else:
            masks_t = torch.zeros((0, H, W), dtype=torch.uint8)
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": torch.tensor([idx]),
            "area": (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
                    if len(boxes) else torch.zeros((0,)),
            "iscrowd": torch.zeros((len(masks),), dtype=torch.int64),
        }
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def selftest():
    train_entries, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)
    assert len(train_entries) + len(val_entries) > 0

    train_files = set(e["file_name"] for e in train_entries)
    val_files = set(e["file_name"] for e in val_entries)
    assert not (train_files & val_files), "train/val file leakage"

    ds = FilamentDataset(train_entries[:2], per_image)
    img, target = ds[0]
    aug_ds = FilamentDataset(train_entries[:2], per_image, augment=True)
    aug_img, aug_target = aug_ds[0]
    assert aug_img.shape == img.shape
    assert img.shape == (3, H, W)
    n = target["masks"].shape[0]
    assert target["boxes"].shape == (n, 4)
    assert target["labels"].shape == (n,)
    if n:
        assert set(target["labels"].tolist()) == {1}
        # box must bound the mask
        m = target["masks"][0].numpy()
        ys, xs = np.where(m)
        x0, y0, x1, y1 = target["boxes"][0].tolist()
        assert x0 <= xs.min() and x1 >= xs.max() + 1
        assert y0 <= ys.min() and y1 >= ys.max() + 1
    print(f"selftest OK (train={len(train_entries)} val={len(val_entries)}, "
          f"sample has {n} filaments)")


if __name__ == "__main__":
    selftest()
