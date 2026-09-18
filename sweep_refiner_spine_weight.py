"""Every experiment this session targeted the DETECTOR stage (which axis
gets a filament flagged at all). This targets the REFINER instead -- the
component that determines mask boundary quality once something's already
detected, genuinely untried all session. The refiner's auxiliary spine
(centerline) supervision head currently gets spine_w=0.3 of the loss
(train_refiner.py); this sweeps that weight to see if leaning on the real,
human-annotated centerline more (or less) sharpens refined masks.

Small model, small 256x256 crops -- trains in minutes locally, not hours.
Saves to distinct filenames so it never touches the deployed
checkpoints/refiner_v5_best.pt.
"""
import time
from pathlib import Path

import torch

from crop_dataset import SpineCropDataset, CROP_SIZE
from train_refiner import RefinerUNet, MaskSpineLoss, ensure_spine_crop_cache

CKPT_DIR = Path("checkpoints")
EPOCHS = 15
LR = 1e-3
BATCH_SIZE = 16


def train_one(spine_w, tr_img, tr_mask, tr_spine, va_img, va_mask, va_spine, device):
    train_ds = SpineCropDataset(tr_img, tr_mask, tr_spine, augment=True)
    val_ds = SpineCropDataset(va_img, va_mask, va_spine, augment=False)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = RefinerUNet(in_channels=1, out_channels=2).to(device)
    criterion = MaskSpineLoss(spine_w=spine_w)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val = float("inf")
    best_state = None
    for epoch in range(EPOCHS):
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

        model.eval()
        val_losses = []
        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(device), mask.to(device)
                val_losses.append(criterion(model(img), mask).item())
        val_loss = sum(val_losses) / len(val_losses)
        print(f"  spine_w={spine_w} epoch {epoch}: train_loss={running/len(train_loader):.4f} "
              f"val_loss={val_loss:.4f} ({time.time()-t0:.0f}s)", flush=True)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {"model": model.state_dict(), "epoch": epoch,
                           "in_channels": 1, "out_channels": 2}

    out_path = CKPT_DIR / f"refiner_spine{str(spine_w).replace('.', '')}_best.pt"
    torch.save(best_state, out_path)
    print(f"  saved {out_path} (best val_loss={best_val:.4f})", flush=True)
    return out_path, best_val


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    tr_img, tr_mask, tr_spine, va_img, va_mask, va_spine = ensure_spine_crop_cache(full_data=False)
    print(f"train crops path: {tr_img}  val crops path: {va_img}", flush=True)

    results = []
    for spine_w in [0.15, 0.3, 0.45, 0.6]:
        print(f"\n=== training spine_w={spine_w} ===", flush=True)
        path, val_loss = train_one(spine_w, tr_img, tr_mask, tr_spine, va_img, va_mask, va_spine, device)
        results.append((spine_w, path, val_loss))

    print("\n=== summary (lower val_loss is better) ===")
    for spine_w, path, val_loss in sorted(results, key=lambda x: x[2]):
        print(f"spine_w={spine_w}: val_loss={val_loss:.4f}  ({path})")


if __name__ == "__main__":
    main()
