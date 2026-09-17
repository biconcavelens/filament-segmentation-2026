"""Real-hint crop cache: pairs each GT filament with the CURRENT detector's
own coarse mask on that instance (not a synthetic degradation of the GT).

Last night's hint-refiner experiment trained on `degrade_mask()` (a synthetic
corruption of the GT) and val_loss improved a lot but real pipeline PQ didn't
move -- diagnosed as train/inference mismatch: synthetic hints don't look like
what the real detector actually produces. Fix: generate hints from the real
detector's real output on train images, so the refiner sees genuine detector
noise (mislocalization, missed barbs, wrong scale) during training too.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import pycocotools.mask as mu

from dataset import train_val_split, IMG_DIR, H, W
from crop_dataset import square_bounds, CROP_SIZE
from train import build_from_checkpoint

CACHE_DIR = Path("real_hint_cache")
DET_SCORE_FOR_CACHE = 0.5  # looser than inference thresh so more GT get a hint pair
MATCH_IOU_MIN = 0.1        # a detection is "for" a GT box if they overlap at all


@torch.no_grad()
def build_cache(entries: list[dict], per_image: dict, detector, device, prefix: str):
    CACHE_DIR.mkdir(exist_ok=True)
    img_path = CACHE_DIR / f"{prefix}_images.npy"
    hint_path = CACHE_DIR / f"{prefix}_hints.npy"
    mask_path = CACHE_DIR / f"{prefix}_masks.npy"
    if img_path.exists():
        print(f"{prefix} cache already exists, skipping")
        return

    imgs, hints, masks = [], [], []
    for i, e in enumerate(entries, 1):
        anns = per_image.get(e["id"], [])
        if not anns:
            continue
        gray = np.array(Image.open(IMG_DIR / e["file_name"]).convert("L"))
        rgb = np.array(Image.open(IMG_DIR / e["file_name"]).convert("RGB"))
        img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0 / 1.0
        out = detector([img_t.to(device)])[0]
        scores = out["scores"].cpu().numpy()
        det_masks = out["masks"].cpu().numpy()
        keep = scores >= DET_SCORE_FOR_CACHE
        det_masks = (det_masks[keep, 0] > 0.5).astype(np.uint8)  # (n_det, H, W)

        gt_masks = []
        for a in anns:
            rles = mu.frPyObjects(a["segmentation"], H, W)
            gt_masks.append(mu.decode(mu.merge(rles)).astype(np.uint8))

        for m in gt_masks:
            if m.sum() == 0:
                continue
            # find best-overlapping detection for this GT instance
            best_iou, best_det = 0.0, None
            for dm in det_masks:
                inter = np.logical_and(m, dm).sum()
                if inter == 0:
                    continue
                union = np.logical_or(m, dm).sum()
                iou = inter / union
                if iou > best_iou:
                    best_iou, best_det = iou, dm
            if best_det is None or best_iou < MATCH_IOU_MIN:
                continue  # no real detection to pair with this GT -> skip

            x0, y0, x1, y1 = square_bounds(m)
            crop_img = np.array(Image.fromarray(gray[y0:y1, x0:x1]).resize(
                (CROP_SIZE, CROP_SIZE), Image.BILINEAR))
            crop_mask = np.array(Image.fromarray(m[y0:y1, x0:x1] * 255).resize(
                (CROP_SIZE, CROP_SIZE), Image.NEAREST)) > 127
            crop_hint = np.array(Image.fromarray(best_det[y0:y1, x0:x1] * 255).resize(
                (CROP_SIZE, CROP_SIZE), Image.NEAREST)) > 127

            imgs.append(crop_img)
            hints.append(crop_hint.astype(np.uint8))
            masks.append(crop_mask.astype(np.uint8))

        if i % 100 == 0:
            print(f"  {i}/{len(entries)} images, {len(imgs)} pairs so far", flush=True)

    np.save(img_path, np.stack(imgs).astype(np.uint8))
    np.save(hint_path, np.stack(hints).astype(np.uint8))
    np.save(mask_path, np.stack(masks).astype(np.uint8))
    print(f"{prefix}: {len(imgs)} real-hint pairs saved to {CACHE_DIR}")


class RealHintCropDataset(torch.utils.data.Dataset):
    """2-channel input: [gray crop, REAL detector coarse mask]. Unlike
    HintCropDataset (synthetic degrade_mask), the hint here was produced by
    actually running the detector, so its noise matches inference exactly."""

    def __init__(self, prefix: str, augment: bool = False):
        self.images = np.load(CACHE_DIR / f"{prefix}_images.npy", mmap_mode="r")
        self.hints = np.load(CACHE_DIR / f"{prefix}_hints.npy", mmap_mode="r")
        self.masks = np.load(CACHE_DIR / f"{prefix}_masks.npy", mmap_mode="r")
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = np.asarray(self.images[idx]).copy()
        hint = np.asarray(self.hints[idx]).copy()
        mask = np.asarray(self.masks[idx]).copy()

        x = torch.stack([torch.from_numpy(img).float() / 255.0,
                         torch.from_numpy(hint).float()], 0)
        y = torch.from_numpy(mask).float().unsqueeze(0)

        if self.augment:
            import random
            k = random.randrange(4)
            x, y = torch.rot90(x, k, (1, 2)), torch.rot90(y, k, (1, 2))
            if random.random() < 0.5:
                x, y = torch.flip(x, (2,)), torch.flip(y, (2,))
            if random.random() < 0.5:
                x[0] = torch.clamp(x[0] * random.uniform(0.85, 1.15), 0, 1)
        return x, y


def main():
    device = torch.device("cuda")
    state = torch.load("checkpoints/maskrcnn_epoch3.pt", map_location=device)
    detector = build_from_checkpoint(state, num_classes=2).to(device)
    detector.load_state_dict(state["model"])
    detector.roi_heads.nms_thresh = 0.30
    detector.eval()

    train_entries, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)
    build_cache(train_entries, per_image, detector, device, "train")
    build_cache(val_entries, per_image, detector, device, "val")


if __name__ == "__main__":
    main()
