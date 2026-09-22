"""Mask2Former (Swin-Tiny, frozen encoder) with a much longer training
budget than the original local attempt (real PQ 0.28) -- that attempt used
only 10 epochs at batch_size=1, explicitly diagnosed as "insufficient
fine-tune budget on ~1k images" in README.md's rejected-approaches table.
Every other architecture this session needed 40 epochs to converge
properly (YOLO, RT-DETR, YOLO26 all use 40); Mask2Former never got that
chance. Everything else identical to the original recipe (frozen Swin
encoder, batch_size=1, lr=1e-4, single-class) so epoch count is the only
new variable.

Architecturally, Mask2Former is a genuine departure from every detector
tried this session (Mask R-CNN, YOLO, RT-DETR, YOLO26): no box-proposal
stage at all. A fixed set of learned queries attend directly to image
features via mask attention, producing instance masks without ever going
through an axis-aligned bounding box. This session's confidence-
miscalibration root cause (diag_missed_filaments.py) was found
independently in every box-based detector tried -- Mask2Former's
mechanism sidesteps that failure mode entirely, for better or worse.

Saves every epoch's checkpoint so the best one can be selected via the
full PQ-sweep pipeline afterward (matching every other detector's
selection methodology this session), not just lowest val_loss (loss and
PQ don't always agree, established repeatedly this session).
"""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import pycocotools.mask as mu
from PIL import Image
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

DATA = Path("/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
IMG_DIR = DATA / "train" / "train_images"
ANN_PATH = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
CKPT_OUT = Path("/kaggle/working/checkpoints")
CKPT_OUT.mkdir(parents=True, exist_ok=True)

H, W = 2048, 2048
INPUT_SIZE = 1024
CHECKPOINT_NAME = "facebook/mask2former-swin-tiny-coco-instance"
EPOCHS = 40
LR = 1e-4
COOLDOWN_EVERY = 200
COOLDOWN_SECONDS = 10


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


class Mask2FormerFilamentDataset(torch.utils.data.Dataset):
    def __init__(self, entries, per_image, processor):
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
        masks = np.stack(masks).astype(np.float32)

        pixel_values = self.processor(img, return_tensors="pt")["pixel_values"][0]

        mask_t = torch.from_numpy(masks).unsqueeze(0)
        mask_t = torch.nn.functional.interpolate(mask_t, size=(INPUT_SIZE, INPUT_SIZE),
                                                   mode="nearest")[0]
        class_labels = torch.zeros(len(masks), dtype=torch.long)
        return pixel_values, mask_t, class_labels


def collate_single(batch):
    assert len(batch) == 1
    return batch[0]


def build_model(device):
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        CHECKPOINT_NAME, num_labels=1, ignore_mismatched_sizes=True
    ).to(device)
    for p in model.model.pixel_level_module.encoder.parameters():
        p.requires_grad = False
    return model


def run_one(model, device, batch):
    pixel_values, mask_labels, class_labels = batch
    pixel_values = pixel_values.unsqueeze(0).to(device)
    out = model(pixel_values=pixel_values,
                mask_labels=[mask_labels.to(device)],
                class_labels=[class_labels.to(device)])
    return out.loss


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = [run_one(model, device, b).item() for b in loader]
    return sum(losses) / len(losses)


def train_with_resume(model, train_loader, val_loader, device, max_retries=6):
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    start_epoch = 0
    last_ckpt = CKPT_OUT / "mask2former_long_last.pt"
    if last_ckpt.exists():
        state = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(state["model"])
        start_epoch = state["epoch"] + 1
        for _ in range(start_epoch):
            scheduler.step()
        print(f"resuming from {last_ckpt} at epoch {start_epoch}", flush=True)

    attempt = 0
    epoch = start_epoch
    while epoch < EPOCHS and attempt <= max_retries:
        try:
            model.train()
            t0 = time.time()
            running = 0.0
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                loss = run_one(model, device, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=0.1)
                optimizer.step()
                running += loss.item()

                if (i + 1) % 100 == 0:
                    print(f"  epoch {epoch} step {i+1}/{len(train_loader)} loss={running/(i+1):.2f}", flush=True)
                if (i + 1) % COOLDOWN_EVERY == 0:
                    torch.cuda.empty_cache()
                    time.sleep(COOLDOWN_SECONDS)
            scheduler.step()

            val_loss = evaluate(model, val_loader, device)
            print(f"epoch {epoch}: train_loss={running/len(train_loader):.2f} "
                  f"val_loss={val_loss:.2f} ({time.time()-t0:.0f}s)", flush=True)

            payload = {"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}
            torch.save(payload, CKPT_OUT / f"mask2former_long_epoch{epoch}.pt")
            torch.save(payload, last_ckpt)
            # keep storage/download size bounded -- Swin-Tiny checkpoints are
            # large enough that 40 of them risks the slow/unreliable large-
            # output downloads seen with other kernels this session
            old = CKPT_OUT / f"mask2former_long_epoch{epoch - 5}.pt"
            if old.exists():
                old.unlink()
            epoch += 1
        except Exception as e:
            attempt += 1
            print(f"training crashed at epoch {epoch} (attempt {attempt}/{max_retries}): {e}", flush=True)
            if attempt > max_retries:
                raise
    print("training finished normally", flush=True)


def main():
    device = torch.device("cuda")
    print("device:", device, "EPOCHS:", EPOCHS, flush=True)

    processor = Mask2FormerImageProcessor.from_pretrained(
        # newer transformers versions require longest_edge strictly > shortest_edge
        # (the original local run predates this validation); +32 keeps Swin's
        # stride-32 divisibility while satisfying the check for our square inputs
        CHECKPOINT_NAME, size={"shortest_edge": INPUT_SIZE, "longest_edge": INPUT_SIZE + 32},
        do_reduce_labels=False,
    )
    train_entries, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)
    train_ds = Mask2FormerFilamentDataset(train_entries, per_image, processor)
    val_ds = Mask2FormerFilamentDataset(val_entries, per_image, processor)
    print(f"train images: {len(train_ds)}  val images: {len(val_ds)}", flush=True)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True,
                                                collate_fn=collate_single, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False,
                                              collate_fn=collate_single, num_workers=0)

    model = build_model(device)
    train_with_resume(model, train_loader, val_loader, device)

    # copy the last few epoch checkpoints out for full PQ-sweep selection locally
    import shutil
    for epoch in range(max(0, EPOCHS - 5), EPOCHS):
        src = CKPT_OUT / f"mask2former_long_epoch{epoch}.pt"
        if src.exists():
            shutil.copy(src, Path("/kaggle/working") / src.name)
    print("copied final checkpoints to /kaggle/working", flush=True)


if __name__ == "__main__":
    main()
