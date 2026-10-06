"""[semseg3: same ConvNeXt-Tiny recipe as the deployed S, but a 5.5h budget (S was still improving at 3h, loss 0.557), seed 2]
Decoupled semantic route (forum hint: a strong segmentation net with focal +
Dice, instances split afterwards). smp U-Net, ImageNet ConvNeXt-Tiny encoder,
trained at NATIVE resolution on 1024px crops of the 2048px images (no
downsampling -- filaments and barbs are a few pixels wide), two heads:
  ch0 filament foreground  -- focal + Dice
  ch1 GT spine (centerline, 5px wide) -- BCE + Dice; watershed seeds for
      splitting touching filaments later
Every (image, annotator) entry is a training sample, same split as
dataset.train_val_split(val_frac=0.1, seed=0). Then full-image 4-flip-TTA
probability maps for the 70 val images and 180 test images, saved as uint8
npz per image stem, zipped as probs.zip (<stem>.npz with arrays fg, spine).

Time-budgeted (TRAIN_HOURS, cosine LR over the budget) so the kernel completes.
"""
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "segmentation-models-pytorch"], check=True)

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = next(Path("/kaggle/input").rglob("MAGFiLO_1.0_Kaggle_2026"))
ANN = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
IMG_DIR, TEST_DIR = DATA / "train" / "train_images", DATA / "test" / "test_images"
OUT = Path("/kaggle/working")
H = W = 2048
CROP, BATCH, LR = 1024, 4, 3e-4
TRAIN_HOURS = float(os.environ.get("SEMSEG_HOURS", "5.5"))
ENCODER = os.environ.get("SEMSEG_ENCODER", "tu-convnext_tiny")
SEED = 2


def load():
    coco = json.load(open(ANN, encoding="utf-8"))
    per_image = {}
    for a in coco["annotations"]:
        per_image.setdefault(a["image_id"], []).append(a)
    images = coco["images"]
    files = sorted(set(i["file_name"] for i in images))
    rng = random.Random(0)
    rng.shuffle(files)
    val_files = set(files[:max(1, int(len(files) * 0.1))])
    train = [i for i in images if i["file_name"] not in val_files]
    return train, sorted(val_files), per_image


def render(anns):
    fg = np.zeros((H, W), np.uint8)
    sp = np.zeros((H, W), np.uint8)
    for a in anns:
        seg = a["segmentation"]
        seg = json.loads(seg) if isinstance(seg, str) else seg
        for poly in seg:
            if len(poly) >= 6:
                cv2.fillPoly(fg, [np.asarray(poly, np.float32).reshape(-1, 2).round().astype(np.int32)], 1)
        spine = a.get("spine")
        spine = json.loads(spine) if isinstance(spine, str) else spine
        if spine and len(spine) >= 4:
            cv2.polylines(sp, [np.asarray(spine, np.float32).reshape(-1, 2).round().astype(np.int32)],
                          False, 1, thickness=5)
    return fg, sp


class Crops(torch.utils.data.Dataset):
    def __init__(self, entries, per_image):
        self.entries, self.per_image = entries, per_image

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        img = cv2.imread(str(IMG_DIR / e["file_name"]), cv2.IMREAD_GRAYSCALE)
        anns = self.per_image.get(e["id"], [])
        fg, sp = render(anns)
        if anns and random.random() < 0.75:  # centre on a random filament most of the time
            a = random.choice(anns)
            bb = a["bbox"]
            bb = json.loads(bb) if isinstance(bb, str) else bb
            cx, cy = bb[0] + bb[2] / 2 + random.uniform(-300, 300), bb[1] + bb[3] / 2 + random.uniform(-300, 300)
        else:
            cx, cy = random.uniform(CROP / 2, W - CROP / 2), random.uniform(CROP / 2, H - CROP / 2)
        x0 = int(np.clip(cx - CROP / 2, 0, W - CROP))
        y0 = int(np.clip(cy - CROP / 2, 0, H - CROP))
        img, fg, sp = (a_[y0:y0 + CROP, x0:x0 + CROP] for a_ in (img, fg, sp))
        k = random.randrange(4)  # full dihedral group
        img, fg, sp = (np.rot90(a_, k) for a_ in (img, fg, sp))
        if random.random() < 0.5:
            img, fg, sp = (a_[:, ::-1] for a_ in (img, fg, sp))
        x = img.astype(np.float32) / 255.0
        disk = img > 8
        x = np.where(disk, np.clip(((x ** random.uniform(0.85, 1.18)) - 0.5) * random.uniform(0.85, 1.15)
                                   + 0.5 + random.uniform(-0.05, 0.05), 0, 1), x)
        return (torch.from_numpy(np.ascontiguousarray(x))[None],
                torch.from_numpy(np.ascontiguousarray(np.stack([fg, sp]))).float())


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = smp.Unet(ENCODER, encoder_weights="imagenet", in_channels=1, classes=2)

    def forward(self, x):
        return self.net((x - 0.45) / 0.225)


def dice_loss(logits, t, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * t).sum((1, 2))
    return (1 - (2 * inter + eps) / (p.sum((1, 2)) + t.sum((1, 2)) + eps)).mean()


def focal_loss(logits, t, gamma=2.0, alpha=0.75):
    bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * t + (1 - p) * (1 - t)
    w = alpha * t + (1 - alpha) * (1 - t)
    return (w * (1 - pt) ** gamma * bce).mean() * 10.0


def loss_fn(out, tgt):
    fg_l, sp_l = out[:, 0], out[:, 1]
    fg_t, sp_t = tgt[:, 0], tgt[:, 1]
    return (focal_loss(fg_l, fg_t) + dice_loss(fg_l, fg_t)
            + 0.5 * (F.binary_cross_entropy_with_logits(sp_l, sp_t) + dice_loss(sp_l, sp_t)))


@torch.no_grad()
def predict_full(model, img_u8, device):
    x = torch.from_numpy(img_u8.astype(np.float32) / 255.0)[None, None].to(device)
    acc = 0
    for flip in [(), (3,), (2,), (2, 3)]:
        xi = torch.flip(x, flip) if flip else x
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.sigmoid(model(xi).float())
        acc = acc + (torch.flip(p, flip) if flip else p)
    p = (acc / 4)[0].cpu().numpy()
    return (p * 255).round().astype(np.uint8)


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda")
    train, val_files, per_image = load()
    print(f"{len(train)} train entries, {len(val_files)} val images, encoder {ENCODER}", flush=True)
    loader = torch.utils.data.DataLoader(Crops(train, per_image), batch_size=BATCH, shuffle=True,
                                         num_workers=4, drop_last=True, persistent_workers=True)
    model = Net().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    budget, t0, step, epoch = TRAIN_HOURS * 3600, time.time(), 0, 0
    done = False
    while not done:
        model.train()
        run, n = 0.0, 0
        for x, y in loader:
            frac = (time.time() - t0) / budget
            if frac >= 1:
                done = True
                break
            for g in opt.param_groups:  # cosine over the time budget, short warmup
                g["lr"] = LR * min(1.0, step / 200) * 0.5 * (1 + math.cos(math.pi * frac))
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x)
            loss = loss_fn(out.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run, n, step = run + loss.item(), n + 1, step + 1
        epoch += 1
        print(f"epoch {epoch}: loss={run / max(n, 1):.4f} step={step} "
              f"elapsed={(time.time() - t0) / 3600:.2f}h", flush=True)
    torch.save({"model": model.state_dict(), "encoder": ENCODER}, OUT / "semseg.pt")

    model.eval()
    (OUT / "probs").mkdir(exist_ok=True)
    stems = [(IMG_DIR / f) for f in val_files] + sorted(TEST_DIR.iterdir())
    for k, path in enumerate(stems, 1):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        p = predict_full(model, img, device)
        np.savez_compressed(OUT / "probs" / f"{path.stem}.npz", fg=p[0], spine=p[1])
        if k % 25 == 0:
            print(f"  predicted {k}/{len(stems)}", flush=True)
    import shutil
    shutil.make_archive(str(OUT / "probs"), "zip", OUT / "probs")  # one file: the CLI only lists a page of outputs
    shutil.rmtree(OUT / "probs")
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
