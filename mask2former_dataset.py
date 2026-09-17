"""Per-image dataset for fine-tuning Mask2Former (single class: filament).

Unlike Mask R-CNN, Mask2Former has no box-proposal stage — a fixed set of
learned queries attend directly to image features via mask attention, which
should handle thin/elongated/diagonal filaments better than an axis-aligned
box paradigm.
"""
import numpy as np
import torch
from PIL import Image
import pycocotools.mask as mu

from dataset import IMG_DIR, H, W

INPUT_SIZE = 1024  # native 2048 halved; Swin needs a size divisible by 32


class Mask2FormerFilamentDataset(torch.utils.data.Dataset):
    def __init__(self, entries: list[dict], per_image: dict, processor):
        self.entries = [e for e in entries if per_image.get(e["id"])]
        self.per_image = per_image
        self.processor = processor

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        anns = self.per_image[entry["id"]]
        img = Image.open(IMG_DIR / entry["file_name"]).convert("RGB")

        masks = []
        for a in anns:
            rles = mu.frPyObjects(a["segmentation"], H, W)
            m = mu.decode(mu.merge(rles))
            if m.sum() == 0:
                continue
            masks.append(m)
        masks = np.stack(masks).astype(np.float32)  # [N,2048,2048]

        pixel_values = self.processor(img, return_tensors="pt")["pixel_values"][0]  # [3,1024,1024]

        mask_t = torch.from_numpy(masks).unsqueeze(0)
        mask_t = torch.nn.functional.interpolate(mask_t, size=(INPUT_SIZE, INPUT_SIZE),
                                                   mode="nearest")[0]  # [N,1024,1024]
        class_labels = torch.zeros(len(masks), dtype=torch.long)  # single class -> id 0
        return pixel_values, mask_t, class_labels


def collate_single(batch):
    assert len(batch) == 1
    return batch[0]


def selftest():
    from transformers import Mask2FormerImageProcessor
    from dataset import train_val_split

    processor = Mask2FormerImageProcessor.from_pretrained(
        "facebook/mask2former-swin-tiny-coco-instance",
        size={"shortest_edge": INPUT_SIZE, "longest_edge": INPUT_SIZE},
        do_reduce_labels=False,
    )
    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)
    ds = Mask2FormerFilamentDataset(val_entries[:3], per_image, processor)
    assert len(ds) > 0

    pixel_values, mask_labels, class_labels = ds[0]
    n = mask_labels.shape[0]
    assert pixel_values.shape == (3, INPUT_SIZE, INPUT_SIZE)
    assert mask_labels.shape == (n, INPUT_SIZE, INPUT_SIZE)
    assert class_labels.shape == (n,)
    assert n > 0 and mask_labels.sum() > 0
    assert (class_labels == 0).all()
    print(f"selftest OK (entry has {n} instances)")


if __name__ == "__main__":
    selftest()
