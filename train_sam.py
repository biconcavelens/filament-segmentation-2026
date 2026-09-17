"""Fine-tune SAM's mask decoder (4.1M params) with box prompts, encoder frozen.

Usage:
    python train_sam.py --epochs 5
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import SamModel, SamProcessor

from sam_dataset import SamFilamentDataset, collate_single
from dataset import train_val_split

CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)
COOLDOWN_EVERY = 200
COOLDOWN_SECONDS = 10


class DiceBCELoss(nn.Module):
    def __init__(self, bce_w=0.5, dice_w=0.5):
        super().__init__()
        self.bce_w, self.dice_w = bce_w, dice_w
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum((1, 2))
        union = probs.sum((1, 2)) + targets.sum((1, 2))
        dice = 1.0 - ((2 * inter + 1e-6) / (union + 1e-6)).mean()
        return self.bce_w * bce + self.dice_w * dice


def build_model(device):
    model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    for p in model.vision_encoder.parameters():
        p.requires_grad = False
    for p in model.prompt_encoder.parameters():
        p.requires_grad = False
    return model


def run_one(model, criterion, device, batch, train: bool):
    pixel_values, input_boxes, masks_gt = batch
    pixel_values = pixel_values.unsqueeze(0).to(device)   # [1,3,1024,1024]
    input_boxes = input_boxes.unsqueeze(0).to(device)     # [N,4] -> [1,N,4]
    masks_gt = masks_gt.to(device)

    with torch.no_grad():
        image_embeddings = model.get_image_embeddings(pixel_values)

    out = model(image_embeddings=image_embeddings, input_boxes=input_boxes, multimask_output=False)
    pred = out.pred_masks.reshape(-1, 256, 256)  # [N,256,256]
    return criterion(pred, masks_gt)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses = [run_one(model, criterion, device, b, train=False).item() for b in loader]
    return sum(losses) / len(losses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    train_entries, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)
    train_ds = SamFilamentDataset(train_entries, per_image, processor)
    val_ds = SamFilamentDataset(val_entries, per_image, processor)
    print(f"train images: {len(train_ds)}  val images: {len(val_ds)}")

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True,
                                                collate_fn=collate_single, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False,
                                              collate_fn=collate_single, num_workers=0)

    model = build_model(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.mask_decoder.parameters(), lr=args.lr)

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss = run_one(model, criterion, device, batch, train=True)
            loss.backward()
            optimizer.step()
            running += loss.item()

            if (i + 1) % 100 == 0:
                print(f"  epoch {epoch} step {i+1}/{len(train_loader)} loss={running/(i+1):.4f}")
            if (i + 1) % COOLDOWN_EVERY == 0:
                torch.cuda.empty_cache()
                time.sleep(COOLDOWN_SECONDS)

        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"epoch {epoch}: train_loss={running/len(train_loader):.4f} "
              f"val_loss={val_loss:.4f} ({time.time()-t0:.0f}s)")

        ckpt = CKPT_DIR / f"sam_decoder_epoch{epoch}.pt"
        torch.save({"mask_decoder": model.mask_decoder.state_dict(), "epoch": epoch}, ckpt)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"mask_decoder": model.mask_decoder.state_dict(), "epoch": epoch},
                       CKPT_DIR / "sam_decoder_best.pt")
            print(f"  new best (val_loss={val_loss:.4f}), saved sam_decoder_best.pt")


if __name__ == "__main__":
    main()
