"""Full-image binary segmentation dataset: is this pixel part of ANY filament.

No instance-awareness here at all -- connected components does that job at
inference time, same idea as the classical baseline but with a learned
discriminator instead of a hand-crafted local-background threshold.
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from PIL import Image
import pycocotools.mask as mu

from dataset import IMG_DIR, H, W

TRAIN_SIZE = 1024

train_transform = A.Compose([
    A.Resize(TRAIN_SIZE, TRAIN_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
                        border_mode=0, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(TRAIN_SIZE, TRAIN_SIZE),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2(),
])


class UnionMaskDataset(torch.utils.data.Dataset):
    def __init__(self, entries: list[dict], per_image: dict, transform):
        self.entries = [e for e in entries if per_image.get(e["id"])]
        self.per_image = per_image
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img = np.array(Image.open(IMG_DIR / entry["file_name"]).convert("L"))

        union = np.zeros((H, W), dtype=np.uint8)
        for a in self.per_image[entry["id"]]:
            rles = mu.frPyObjects(a["segmentation"], H, W)
            union |= mu.decode(mu.merge(rles))

        out = self.transform(image=img, mask=union.astype(np.float32))
        img_t = out["image"]
        mask_t = out["mask"].unsqueeze(0).float()
        return img_t, mask_t


def selftest():
    from dataset import train_val_split

    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)
    ds = UnionMaskDataset(val_entries[:3], per_image, val_transform)
    assert len(ds) > 0

    img_t, mask_t = ds[0]
    assert img_t.shape == (1, TRAIN_SIZE, TRAIN_SIZE)
    assert mask_t.shape == (1, TRAIN_SIZE, TRAIN_SIZE)
    assert set(mask_t.unique().tolist()) <= {0.0, 1.0}
    assert mask_t.sum() > 0, "expected non-empty union mask"
    print(f"selftest OK (img range [{img_t.min():.2f}, {img_t.max():.2f}])")


if __name__ == "__main__":
    selftest()
