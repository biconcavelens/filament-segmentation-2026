"""Fine-tune Mask2Former (Swin-Tiny backbone frozen) for filament instance segmentation.

Usage:
    python train_mask2former.py --epochs 10
"""
import argparse
import time
from pathlib import Path

import torch
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

from mask2former_dataset import Mask2FormerFilamentDataset, collate_single, INPUT_SIZE
from dataset import train_val_split

CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)
COOLDOWN_EVERY = 200
COOLDOWN_SECONDS = 10
CHECKPOINT_NAME = "facebook/mask2former-swin-tiny-coco-instance"


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    processor = Mask2FormerImageProcessor.from_pretrained(
        # newer transformers versions require longest_edge strictly > shortest_edge
        CHECKPOINT_NAME, size={"shortest_edge": INPUT_SIZE, "longest_edge": INPUT_SIZE + 32},
        do_reduce_labels=False,
    )
    train_entries, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)
    train_ds = Mask2FormerFilamentDataset(train_entries, per_image, processor)
    val_ds = Mask2FormerFilamentDataset(val_entries, per_image, processor)
    print(f"train images: {len(train_ds)}  val images: {len(val_ds)}")

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True,
                                                collate_fn=collate_single, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False,
                                              collate_fn=collate_single, num_workers=0)

    model = build_model(device)
    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        start_epoch = state["epoch"] + 1
        best_val = state.get("val_loss", float("inf"))
        print(f"resumed from {args.resume} at epoch {start_epoch} (best_val={best_val})")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    for _ in range(start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, args.epochs):
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
                print(f"  epoch {epoch} step {i+1}/{len(train_loader)} loss={running/(i+1):.2f}")
            if (i + 1) % COOLDOWN_EVERY == 0:
                torch.cuda.empty_cache()
                time.sleep(COOLDOWN_SECONDS)
        scheduler.step()

        val_loss = evaluate(model, val_loader, device)
        print(f"epoch {epoch}: train_loss={running/len(train_loader):.2f} "
              f"val_loss={val_loss:.2f} ({time.time()-t0:.0f}s)")

        ckpt = CKPT_DIR / f"mask2former_epoch{epoch}.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, ckpt)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                       CKPT_DIR / "mask2former_best.pt")
            print(f"  new best (val_loss={val_loss:.2f}), saved mask2former_best.pt")


if __name__ == "__main__":
    main()
