"""Train the crop-refiner U-Net on per-instance 256x256 filament crops.

Usage:
    python train_refiner.py --epochs 15
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from crop_dataset import CropDataset, HintCropDataset, build_crop_cache, CROP_SIZE
from dataset import train_val_split

CACHE_DIR = Path("crop_cache")
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)


class ConvBlock(nn.Sequential):
    def __init__(self, in_c, out_c):
        super().__init__(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )


class RefinerUNet(nn.Module):
    """Small from-scratch U-Net; crops are only 256x256 so this trains fast
    and needs no separate pretrained-encoder download."""

    def __init__(self, features=32, in_channels=1, out_channels=1):
        super().__init__()
        f = features
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.enc1 = ConvBlock(in_channels, f)
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
        self.head = nn.Conv2d(f, out_channels, 1)

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


class DiceBCELoss(nn.Module):
    def __init__(self, bce_w=0.5, dice_w=0.5):
        super().__init__()
        self.bce_w, self.dice_w = bce_w, dice_w
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum((2, 3))
        union = probs.sum((2, 3)) + targets.sum((2, 3))
        dice = 1.0 - ((2 * inter + 1e-6) / (union + 1e-6)).mean()
        return self.bce_w * bce + self.dice_w * dice


class MaskSpineLoss(nn.Module):
    """logits: (B,2,H,W) -- channel 0 predicts the mask (DiceBCE, the real
    target), channel 1 predicts the GT spine centerline (plain BCE, purely
    auxiliary -- ignored at inference). The spine term only has to nudge
    the shared encoder/decoder features toward the true centerline; it
    never has to stand on its own the way the mask does."""

    def __init__(self, spine_w=0.3):
        super().__init__()
        self.mask_loss = DiceBCELoss()
        self.spine_bce = nn.BCEWithLogitsLoss()
        self.spine_w = spine_w

    def forward(self, logits, targets):
        mask_logits, spine_logits = logits[:, 0:1], logits[:, 1:2]
        mask_t, spine_t = targets[:, 0:1], targets[:, 1:2]
        return (1 - self.spine_w) * self.mask_loss(mask_logits, mask_t) + \
            self.spine_w * self.spine_bce(spine_logits, spine_t)


def _soft_erode(img):
    p1 = -nn.functional.max_pool2d(-img, (3, 1), stride=1, padding=(1, 0))
    p2 = -nn.functional.max_pool2d(-img, (1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img):
    return nn.functional.max_pool2d(img, 3, stride=1, padding=1)


def _soft_open(img):
    return _soft_dilate(_soft_erode(img))


def _soft_skeleton(img, iters=5):
    """Differentiable morphological skeleton (Shit et al. 2021, clDice), via
    repeated soft erosion: at each step, whatever erosion removes that isn't
    recovered by a matching dilation (soft_open) is skeleton."""
    skel = nn.functional.relu(img - _soft_open(img))
    for _ in range(iters):
        img = _soft_erode(img)
        delta = nn.functional.relu(img - _soft_open(img))
        skel = skel + nn.functional.relu(delta - skel * delta)
    return skel


class DiceBCEClDiceLoss(nn.Module):
    """DiceBCE + clDice: a topology-preserving term for thin tubular
    structures. Standard Dice/BCE score a thin filament fine even when its
    predicted mask is broken into disconnected fragments (small pixel-area
    cost); clDice instead measures overlap along each mask's *skeleton*
    against the other mask's *area*, which collapses to near-zero the moment
    a prediction's centerline strays off the true filament or breaks -- a
    connectivity error a plain per-pixel loss barely notices."""

    def __init__(self, bce_w=0.4, dice_w=0.4, cldice_w=0.2, skel_iters=5):
        super().__init__()
        self.bce_w, self.dice_w, self.cldice_w = bce_w, dice_w, cldice_w
        self.skel_iters = skel_iters
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum((2, 3))
        union = probs.sum((2, 3)) + targets.sum((2, 3))
        dice = 1.0 - ((2 * inter + 1e-6) / (union + 1e-6)).mean()

        skel_pred = _soft_skeleton(probs, self.skel_iters)
        skel_true = _soft_skeleton(targets, self.skel_iters)
        tprec = (torch.sum(skel_pred * targets) + 1.0) / (torch.sum(skel_pred) + 1.0)
        tsens = (torch.sum(skel_true * probs) + 1.0) / (torch.sum(skel_true) + 1.0)
        cldice = 1.0 - 2.0 * (tprec * tsens) / (tprec + tsens)

        return self.bce_w * bce + self.dice_w * dice + self.cldice_w * cldice


class MaskSpineClDiceLoss(nn.Module):
    """MaskSpineLoss (v5, the deployed best) with clDice added to the mask
    term instead of replacing spine supervision -- tests clDice as an
    addition on top of the validated recipe, not instead of it."""

    def __init__(self, spine_w=0.3):
        super().__init__()
        self.mask_loss = DiceBCEClDiceLoss()  # bce_w=0.4, dice_w=0.4, cldice_w=0.2
        self.spine_bce = nn.BCEWithLogitsLoss()
        self.spine_w = spine_w

    def forward(self, logits, targets):
        mask_logits, spine_logits = logits[:, 0:1], logits[:, 1:2]
        mask_t, spine_t = targets[:, 0:1], targets[:, 1:2]
        return (1 - self.spine_w) * self.mask_loss(mask_logits, mask_t) + \
            self.spine_w * self.spine_bce(spine_logits, spine_t)


def ensure_crop_cache():
    train_entries, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)
    train_img = CACHE_DIR / "train_images.npy"
    if not train_img.exists():
        print("building crop cache (one-time)...")
        build_crop_cache(train_entries, per_image, CACHE_DIR, "train")
        build_crop_cache(val_entries, per_image, CACHE_DIR, "val")
    return (CACHE_DIR / "train_images.npy", CACHE_DIR / "train_masks.npy",
            CACHE_DIR / "val_images.npy", CACHE_DIR / "val_masks.npy")


SPINE_CACHE_DIR = Path("spine_crop_cache")
SPINE_CACHE_DIR_FULL = Path("spine_crop_cache_full")


def ensure_spine_crop_cache(full_data: bool = False):
    from crop_dataset import build_spine_crop_cache
    cache_dir = SPINE_CACHE_DIR_FULL if full_data else SPINE_CACHE_DIR
    val_frac = 0 if full_data else 0.1
    train_entries, val_entries, per_image = train_val_split(val_frac=val_frac, seed=0)
    train_img = cache_dir / "train_images.npy"
    if not train_img.exists():
        print("building spine crop cache (one-time)...")
        build_spine_crop_cache(train_entries, per_image, cache_dir, "train")
        if val_entries:
            build_spine_crop_cache(val_entries, per_image, cache_dir, "val")
    if full_data:
        return (cache_dir / "train_images.npy", cache_dir / "train_masks.npy",
                cache_dir / "train_spines.npy", None, None, None)
    return tuple(cache_dir / f"{split}_{kind}.npy"
                 for split in ("train", "val") for kind in ("images", "masks", "spines"))


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
    p.add_argument("--hint", action="store_true",
                    help="2-channel input: gray crop + coarse-mask hint (refiner v2)")
    p.add_argument("--cldice", action="store_true",
                    help="add a topology-preserving clDice term to the loss (refiner v4)")
    p.add_argument("--spine", action="store_true",
                    help="auxiliary output head supervised by the real GT spine centerline "
                         "(host-confirmed fair game); mask-only at inference (refiner v5)")
    p.add_argument("--full-data", action="store_true",
                    help="final refit on all crops (no held-out val); only valid with --spine, "
                         "for the already-validated v5 recipe")
    args = p.parse_args()
    in_ch = 2 if args.hint else 1
    out_ch = 2 if args.spine else 1
    if args.spine and args.cldice:
        prefix = "refiner_v6_full" if args.full_data else "refiner_v6"
    elif args.spine:
        prefix = "refiner_v5_full" if args.full_data else "refiner_v5"
    elif args.cldice:
        prefix = "refiner_v4_hint" if args.hint else "refiner_v4"
    else:
        prefix = "refiner_v2" if args.hint else "refiner"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "in_channels:", in_ch, "out_channels:", out_ch,
          "full_data:", args.full_data)

    if args.spine:
        from crop_dataset import SpineCropDataset
        tr_img, tr_mask, tr_spine, va_img, va_mask, va_spine = ensure_spine_crop_cache(args.full_data)
        train_ds = SpineCropDataset(tr_img, tr_mask, tr_spine, augment=True)
        val_ds = SpineCropDataset(va_img, va_mask, va_spine, augment=False) if va_img else None
    else:
        DS = HintCropDataset if args.hint else CropDataset
        train_img, train_mask, val_img, val_mask = ensure_crop_cache()
        train_ds = DS(train_img, train_mask, augment=True)
        val_ds = DS(val_img, val_mask, augment=False)
    print(f"train crops: {len(train_ds)}  val crops: {len(val_ds) if val_ds else 0}")

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size,
                                                shuffle=True, num_workers=0)
    val_loader = (torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size,
                                               shuffle=False, num_workers=0)
                  if val_ds else [])

    model = RefinerUNet(in_channels=in_ch, out_channels=out_ch).to(device)
    if args.spine and args.cldice:
        criterion = MaskSpineClDiceLoss()
    elif args.spine:
        criterion = MaskSpineLoss()
    elif args.cldice:
        criterion = DiceBCEClDiceLoss()
    else:
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

        val_loss = evaluate(model, val_loader, criterion, device) if val_loader else float("nan")
        print(f"epoch {epoch}: train_loss={running/len(train_loader):.4f} "
              f"val_loss={val_loss:.4f} ({time.time()-t0:.0f}s)")

        payload = {"model": model.state_dict(), "epoch": epoch,
                   "in_channels": in_ch, "out_channels": out_ch}
        torch.save(payload, CKPT_DIR / f"{prefix}_epoch{epoch}.pt")
        if val_loader and val_loss < best_val:
            best_val = val_loss
            torch.save(payload, CKPT_DIR / f"{prefix}_best.pt")
            print(f"  new best (val_loss={val_loss:.4f}), saved {prefix}_best.pt")
        elif not val_loader:
            torch.save(payload, CKPT_DIR / f"{prefix}_best.pt")  # no val signal: last epoch wins


if __name__ == "__main__":
    main()
