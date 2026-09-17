"""Train a dense semantic + distance-transform model for watershed-based
instance segmentation (see watershed_dataset.py for the rationale: a
proposal-free alternative to the Mask R-CNN family, aimed at total-miss
recall and touching-instance separation without any box/NMS machinery).

Usage:
    python train_watershed.py --epochs 20
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from watershed_dataset import WatershedDataset

CACHE_DIR = Path("watershed_cache")
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)


class ConvBlock(nn.Sequential):
    def __init__(self, in_c, out_c):
        super().__init__(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )


class WatershedUNet(nn.Module):
    """2 output channels: [0]=semantic (BCE), [1]=distance-to-boundary (masked MSE, sigmoid-bounded)."""

    def __init__(self, features=32):
        super().__init__()
        f = features
        self.enc1 = ConvBlock(1, f)
        self.enc2 = ConvBlock(f, f * 2)
        self.enc3 = ConvBlock(f * 2, f * 4)
        self.enc4 = ConvBlock(f * 4, f * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(f * 8, f * 16)
        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, 2, 2)
        self.dec4 = ConvBlock(f * 16, f * 8)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, 2)
        self.dec3 = ConvBlock(f * 8, f * 4)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, 2)
        self.dec2 = ConvBlock(f * 4, f * 2)
        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, 2)
        self.dec1 = ConvBlock(f * 2, f)
        self.head = nn.Conv2d(f, 2, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


class SemanticDistanceLoss(nn.Module):
    def __init__(self, sem_w=0.5, dist_w=0.5):
        super().__init__()
        self.sem_w, self.dist_w = sem_w, dist_w
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        sem_logit, dist_logit = logits[:, 0:1], logits[:, 1:2]
        sem_t, dist_t = target[:, 0:1], target[:, 1:2]
        sem_loss = self.bce(sem_logit, sem_t)
        dist_pred = torch.sigmoid(dist_logit)
        mask = sem_t
        dist_loss = ((dist_pred - dist_t) ** 2 * mask).sum() / (mask.sum() + 1.0)
        return self.sem_w * sem_loss + self.dist_w * dist_loss


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        losses.append(criterion(model(x), y).item())
    return sum(losses) / len(losses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--crop", type=int, default=384)
    p.add_argument("--crops-per-image", type=int, default=6)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_ds = WatershedDataset(CACHE_DIR, "train", crop=args.crop,
                                 crops_per_image=args.crops_per_image, augment=True)
    val_ds = WatershedDataset(CACHE_DIR, "val", crop=args.crop,
                               crops_per_image=args.crops_per_image, augment=False)
    print(f"train samples: {len(train_ds)}  val samples: {len(val_ds)}")

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                                shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size,
                                              shuffle=False, num_workers=0)

    model = WatershedUNet().to(device)
    criterion = SemanticDistanceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()

        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"epoch {epoch}: train_loss={running/len(train_loader):.4f} "
              f"val_loss={val_loss:.4f} ({time.time()-t0:.0f}s)", flush=True)

        payload = {"model": model.state_dict(), "epoch": epoch}
        torch.save(payload, CKPT_DIR / f"watershed_epoch{epoch}.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(payload, CKPT_DIR / "watershed_best.pt")
            print(f"  new best (val_loss={val_loss:.4f}), saved watershed_best.pt")


if __name__ == "__main__":
    main()
