"""Per-image (boxes, GT masks) dataset for fine-tuning SAM's mask decoder.

SAM decodes all boxes for one image in a single call after one (frozen)
encoder pass, so each dataset item is a whole image's worth of instances,
not a single crop.
"""
import numpy as np
import torch
from PIL import Image
import pycocotools.mask as mu

from dataset import IMG_DIR, H, W

DECODER_RES = 256  # SAM's native low-res mask output size, used for the loss


class SamFilamentDataset(torch.utils.data.Dataset):
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

        boxes, masks_lowres = [], []
        for a in anns:
            rles = mu.frPyObjects(a["segmentation"], H, W)
            m = mu.decode(mu.merge(rles)).astype(np.uint8)
            if m.sum() == 0:
                continue
            ys, xs = np.where(m)
            boxes.append([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])
            m_lr = np.array(Image.fromarray(m * 255).resize(
                (DECODER_RES, DECODER_RES), Image.NEAREST)) > 127
            masks_lowres.append(m_lr.astype(np.float32))

        inputs = self.processor(img, input_boxes=[boxes], return_tensors="pt")
        pixel_values = inputs["pixel_values"][0]       # [3,1024,1024]
        input_boxes = inputs["input_boxes"][0]         # [N,4]
        masks_t = torch.from_numpy(np.stack(masks_lowres))  # [N,256,256]
        return pixel_values, input_boxes, masks_t


def collate_single(batch):
    assert len(batch) == 1, "SamFilamentDataset is meant to be used with batch_size=1"
    return batch[0]


def selftest():
    from transformers import SamProcessor
    from dataset import train_val_split

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)
    ds = SamFilamentDataset(val_entries[:3], per_image, processor)
    assert len(ds) > 0

    pixel_values, input_boxes, masks = ds[0]
    n = input_boxes.shape[0]
    assert pixel_values.shape == (3, 1024, 1024)
    assert input_boxes.shape == (n, 4)
    assert masks.shape == (n, DECODER_RES, DECODER_RES)
    assert n > 0
    assert masks.sum() > 0, "expected at least one non-empty GT mask"
    print(f"selftest OK (entry has {n} instances)")


if __name__ == "__main__":
    selftest()
