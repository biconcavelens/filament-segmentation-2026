"""Train the refiner on REAL detector-produced hints (see real_hint_dataset.py
for why: last night's synthetic-degrade hint-refiner didn't move real PQ
because degrade_mask() doesn't match what the detector actually gets wrong).

Usage:
    python real_hint_dataset.py       # build the cache once (needs GPU + detector)
    python train_real_hint.py --epochs 15
"""
import argparse
import time
from pathlib import Path

import torch

from train_refiner import RefinerUNet, DiceBCELoss
from real_hint_dataset import RealHintCropDataset

CKPT_DIR = Path("checkpoints")


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        losses.append(criterion(model(img), mask).item())
    return sum(losses) / len(losses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = RealHintCropDataset("train", augment=True)
    val_ds = RealHintCropDataset("val", augment=False)
    print(f"train pairs: {len(train_ds)}  val pairs: {len(val_ds)}")

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                                shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size,
                                              shuffle=False, num_workers=0)

    model = RefinerUNet(in_channels=2).to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            loss = criterion(model(img), mask)
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()

        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"epoch {epoch}: train_loss={running/len(train_loader):.4f} "
              f"val_loss={val_loss:.4f} ({time.time()-t0:.0f}s)", flush=True)

        payload = {"model": model.state_dict(), "epoch": epoch, "in_channels": 2}
        torch.save(payload, CKPT_DIR / f"refiner_v3_epoch{epoch}.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(payload, CKPT_DIR / "refiner_v3_best.pt")
            print(f"  new best (val_loss={val_loss:.4f}), saved refiner_v3_best.pt")


if __name__ == "__main__":
    main()
