"""Train a U-Net (ResNet34, ImageNet-pretrained) for full-image binary
filament segmentation. Same recipe as the community's real 0.57 notebook,
just at higher resolution (1024 vs 512) since this architecture is light.

Usage:
    python train_unet.py --epochs 30
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from unet_dataset import UnionMaskDataset, train_transform, val_transform
from dataset import train_val_split

CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

# ponytail: cooldown exists only because this laptop has been crashing
# repeatedly under sustained load today; drop it on stable hardware.
COOLDOWN_EVERY = 150
COOLDOWN_SECONDS = 8


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(logits.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        inter = (probs * targets).sum(1)
        dice = (2 * inter + self.smooth) / (probs.sum(1) + targets.sum(1) + self.smooth)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return 0.5 * self.bce(logits, targets) + 0.5 * self.dice(logits, targets)


def dice_score(logits, targets, threshold=0.5, smooth=1e-6):
    preds = (torch.sigmoid(logits) > threshold).float().view(logits.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (preds * targets).sum(1)
    return ((2 * inter + smooth) / (preds.sum(1) + targets.sum(1) + smooth)).mean()


def build_model():
    return smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                     in_channels=1, classes=1, activation=None)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses, dices = [], []
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        losses.append(criterion(logits, masks).item())
        dices.append(dice_score(logits, masks).item())
    return sum(losses) / len(losses), sum(dices) / len(dices)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_entries, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)
    train_ds = UnionMaskDataset(train_entries, per_image, train_transform)
    val_ds = UnionMaskDataset(val_entries, per_image, val_transform)
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                                shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size,
                                              shuffle=False, num_workers=2)

    model = build_model().to(device)
    start_epoch = 0
    best_val_dice = 0.0
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        start_epoch = state["epoch"] + 1
        best_val_dice = state.get("val_dice", 0.0)
        print(f"resumed from {args.resume} at epoch {start_epoch} (best_val_dice={best_val_dice:.4f})")

    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    for _ in range(start_epoch):
        scheduler.step()

    epochs_no_improve = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for i, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward()
            optimizer.step()
            running += loss.item()
            if (i + 1) % COOLDOWN_EVERY == 0:
                torch.cuda.empty_cache()
                time.sleep(COOLDOWN_SECONDS)
        scheduler.step()

        val_loss, val_dice = evaluate(model, val_loader, criterion, device)
        print(f"epoch {epoch}: train_loss={running/len(train_loader):.4f} "
              f"val_loss={val_loss:.4f} val_dice={val_dice:.4f} ({time.time()-t0:.0f}s)")

        ckpt = CKPT_DIR / f"unet_epoch{epoch}.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": val_dice}, ckpt)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            epochs_no_improve = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": val_dice},
                       CKPT_DIR / "unet_best.pt")
            print(f"  new best (val_dice={val_dice:.4f}), saved unet_best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    print(f"\nbest val dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    main()
