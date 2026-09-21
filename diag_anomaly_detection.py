"""Cheap diagnostic for the VAE/autoencoder anomaly-detection idea (flagged
earlier as a multi-hour, real-risk undertaking) BEFORE committing to a full
candidate-generation pipeline: train a small convolutional autoencoder on
"normal" quiet-sun texture (patches that don't overlap any GT filament),
then check whether its reconstruction error actually separates filament
pixels from background on the held-out val split. If reconstruction error
doesn't meaningfully correlate with GT filament location, the whole
approach is dead on arrival and not worth building further -- mirrors the
diagnose-before-commit pattern used all session (diag_missed_filaments.py
before the cls-weight fix).

Known risk being tested directly: autoencoders reconstruct blurry averages
and may fail to flag exactly the faint, thin filaments we most need to
catch (the detector-recall bottleneck established this session).
"""
import random

import numpy as np
import torch
import torch.nn as nn
import pycocotools.mask as mu
from PIL import Image

from dataset import train_val_split, IMG_DIR, H, W

PATCH = 64
STRIDE = 32
DEVICE = torch.device("cuda")


class TinyAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),   # 32
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 8
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(inplace=True),  # 16
            nn.ConvTranspose2d(32, 16, 2, stride=2), nn.ReLU(inplace=True),  # 32
            nn.ConvTranspose2d(16, 1, 2, stride=2), nn.Sigmoid(),            # 64
        )

    def forward(self, x):
        return self.dec(self.enc(x))


def gt_mask_for(entry, per_image):
    m = np.zeros((H, W), dtype=np.uint8)
    for a in per_image.get(entry["id"], []):
        rles = mu.frPyObjects(a["segmentation"], H, W)
        m |= mu.decode(mu.merge(rles))
    return m


def collect_background_patches(entries, per_image, n_target, rng):
    """Random patches whose GT mask is entirely 0 (no filament pixels)."""
    patches = []
    entries = list(entries)
    rng.shuffle(entries)
    for e in entries:
        if len(patches) >= n_target:
            break
        gray = np.array(Image.open(IMG_DIR / e["file_name"]).convert("L")).astype(np.float32) / 255.0
        gtm = gt_mask_for(e, per_image)
        tries = 0
        got = 0
        while got < 30 and tries < 200:
            tries += 1
            y0 = rng.randint(0, H - PATCH)
            x0 = rng.randint(0, W - PATCH)
            if gtm[y0:y0 + PATCH, x0:x0 + PATCH].sum() > 0:
                continue
            patches.append(gray[y0:y0 + PATCH, x0:x0 + PATCH])
            got += 1
    return np.stack(patches[:n_target])


def main():
    rng = random.Random(0)
    train_entries, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)

    print("collecting background-only training patches...", flush=True)
    train_patches = collect_background_patches(train_entries, per_image, 8000, rng)
    print(f"collected {len(train_patches)} background patches", flush=True)

    model = TinyAE().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.from_numpy(train_patches).unsqueeze(1).to(DEVICE)
    n = x.shape[0]
    batch = 64
    for epoch in range(10):
        perm = torch.randperm(n)
        running = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = x[idx]
            opt.zero_grad()
            out = model(xb)
            loss = nn.functional.mse_loss(out, xb)
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
        print(f"epoch {epoch}: train_mse={running/n:.5f}", flush=True)

    print("\nscoring val images for filament-vs-background reconstruction error...", flush=True)
    model.eval()
    filament_errs = []
    background_errs = []
    with torch.no_grad():
        for i, e in enumerate(val_entries, 1):
            gray = np.array(Image.open(IMG_DIR / e["file_name"]).convert("L")).astype(np.float32) / 255.0
            gtm = gt_mask_for(e, per_image)
            if gtm.sum() == 0:
                continue

            patches = []
            coords = []
            for y0 in range(0, H - PATCH, STRIDE):
                for x0 in range(0, W - PATCH, STRIDE):
                    patches.append(gray[y0:y0 + PATCH, x0:x0 + PATCH])
                    coords.append((y0, x0))
            patches = np.stack(patches)
            xb = torch.from_numpy(patches).unsqueeze(1).to(DEVICE)
            errs = []
            for j in range(0, len(xb), 256):
                out = model(xb[j:j + 256])
                per_patch_mse = ((out - xb[j:j + 256]) ** 2).mean(dim=(1, 2, 3))
                errs.append(per_patch_mse.cpu().numpy())
            errs = np.concatenate(errs)

            for (y0, x0), err in zip(coords, errs):
                patch_gt_frac = gtm[y0:y0 + PATCH, x0:x0 + PATCH].mean()
                if patch_gt_frac > 0.3:
                    filament_errs.append(err)
                elif patch_gt_frac == 0:
                    background_errs.append(err)

            if i % 20 == 0:
                print(f"  {i}/{len(val_entries)}", flush=True)

    filament_errs = np.array(filament_errs)
    background_errs = np.array(background_errs)
    print(f"\nfilament-heavy patches: {len(filament_errs)}, mean_err={filament_errs.mean():.5f}")
    print(f"background patches:     {len(background_errs)}, mean_err={background_errs.mean():.5f}")
    print(f"ratio (filament/background): {filament_errs.mean() / background_errs.mean():.3f}")

    all_errs = np.concatenate([filament_errs, background_errs])
    all_labels = np.concatenate([np.ones(len(filament_errs)), np.zeros(len(background_errs))])
    order = np.argsort(-all_errs)
    sorted_labels = all_labels[order]
    n_pos = all_labels.sum()
    n_neg = len(all_labels) - n_pos
    tps = np.cumsum(sorted_labels)
    fps = np.cumsum(1 - sorted_labels)
    tpr = tps / n_pos
    fpr = fps / n_neg
    auc = np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr)
    print(f"AUC (reconstruction error as filament-vs-background classifier): {auc:.4f}")
    print("(0.5 = no signal, 1.0 = perfect separation)")


if __name__ == "__main__":
    main()
