"""Self-supervised (SimSiam) continued pretraining of Mask R-CNN's ResNet-50
backbone on unlabeled GONG H-alpha images (download_gong.py).

No annotations are used -- the host confirmed public GONG H-alpha images
without MAGFiLO annotations may be used for self-supervised pretraining.
Starts from the COCO-trained Mask R-CNN v2 backbone (not from scratch) and
adapts it to H-alpha imagery; train_maskrcnn_hires.py --backbone then
fine-tunes the full detector on the labelled competition data as before.

SimSiam: two random views of the same crop -> encoder -> projector ->
predictor; negative cosine similarity with stop-gradient. Works at the
small batch sizes an 8GB GPU allows (no negatives, no momentum encoder).

Views are native-resolution crops from inside the solar disk: NSO's public
JPGs carry text/logo overlays in the corners that the competition images
don't, so everything beyond DISK_MASK_R px from the centre is blacked out.
Augmentations follow the real nuisance variation: blur (seeing), brightness/
contrast (site/exposure), flips and 90-degree rotations (orientation).

    python pretrain_ssl.py --epochs 40
    python pretrain_ssl.py --epochs 1 --max-steps 20      # smoke test
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights

GONG_DIR = Path("data/gong_unlabeled")
OUT = Path("checkpoints")
DISK_MASK_R = 1010      # disk is remapped to radius 900 px; overlays sit beyond ~1000
CROP = 384              # native-resolution crop side (px) ...
VIEW = 256              # ... resized to this for the encoder
CROPS_PER_IMAGE = 4     # samples per image per epoch
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # the backbone's ImageNet/COCO normalisation


def disk_mask(size=2048, r=DISK_MASK_R):
    yy, xx = np.mgrid[:size, :size]
    return ((xx - size / 2) ** 2 + (yy - size / 2) ** 2) <= r ** 2


class GongViews(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = files
        self.mask = disk_mask()

    def __len__(self):
        return len(self.files) * CROPS_PER_IMAGE

    def _load(self, path):
        g = np.array(Image.open(path).convert("L"))
        g[~self.mask] = 0
        return g

    def _crop_center(self, rng):
        # a crop centre inside the disk (radius 900 minus half a crop), so views are mostly on-disk
        while True:
            x, y = rng.uniform(CROP / 2, 2048 - CROP / 2, 2)
            if (x - 1024) ** 2 + (y - 1024) ** 2 <= (900 - CROP / 4) ** 2:
                return int(x), int(y)

    def _view(self, g, cx, cy):
        s = int(CROP * random.uniform(0.6, 1.0))  # scale jitter
        x0 = min(max(cx - s // 2 + random.randint(-32, 32), 0), 2048 - s)
        y0 = min(max(cy - s // 2 + random.randint(-32, 32), 0), 2048 - s)
        im = Image.fromarray(g[y0:y0 + s, x0:x0 + s]).resize((VIEW, VIEW), Image.BILINEAR)
        if random.random() < 0.5:
            im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 2.0)))  # atmospheric seeing
        a = np.asarray(im, np.float32) / 255.0
        a = np.clip((a - a.mean()) * random.uniform(0.7, 1.3) + a.mean() + random.uniform(-0.1, 0.1), 0, 1)
        a = np.rot90(a, random.randrange(4))
        if random.random() < 0.5:
            a = a[:, ::-1]
        t = torch.from_numpy(np.ascontiguousarray(a)).unsqueeze(0).repeat(3, 1, 1)
        return (t - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]

    def __getitem__(self, idx):
        g = self._load(self.files[idx // CROPS_PER_IMAGE])
        cx, cy = self._crop_center(np.random.default_rng())
        return self._view(g, cx, cy), self._view(g, cx, cy)


class SimSiam(nn.Module):
    def __init__(self, body, dim=2048, pred_dim=512):
        super().__init__()
        self.body = body
        self.projector = nn.Sequential(
            nn.Linear(2048, dim, bias=False), nn.BatchNorm1d(dim), nn.ReLU(inplace=True),
            nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim), nn.ReLU(inplace=True),
            nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim, affine=False))
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim, bias=False), nn.BatchNorm1d(pred_dim), nn.ReLU(inplace=True),
            nn.Linear(pred_dim, dim))

    def encode(self, x):
        return self.projector(F.adaptive_avg_pool2d(self.body(x)["3"], 1).flatten(1))

    def forward(self, x1, x2):
        z1, z2 = self.encode(x1), self.encode(x2)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        return -(F.cosine_similarity(p1, z2.detach()).mean() + F.cosine_similarity(p2, z1.detach()).mean()) / 2


def coco_backbone_body():
    m = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1, trainable_backbone_layers=5)
    return m.backbone.body


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--lr", type=float, default=0.01, help="lower than SimSiam's 0.05*bs/256: continued, not scratch")
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--name", default="ssl_simsiam_r50")
    args = p.parse_args()

    files = sorted(GONG_DIR.glob("*.jpg"))
    device = torch.device("cuda")
    loader = torch.utils.data.DataLoader(GongViews(files), batch_size=args.batch_size, shuffle=True,
                                         num_workers=4, drop_last=True, persistent_workers=True)
    model = SimSiam(coco_backbone_body()).to(device)
    params = [{"params": model.body.parameters()}, {"params": model.projector.parameters()},
              {"params": model.predictor.parameters(), "fix_lr": True}]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    print(f"{len(files)} images, {len(loader)} steps/epoch, bs={args.batch_size}, lr={args.lr}", flush=True)

    for epoch in range(args.epochs):
        model.train()
        t0, run, n = time.time(), 0.0, 0
        for i, (v1, v2) in enumerate(loader):
            if args.max_steps and i >= args.max_steps:
                break
            with torch.amp.autocast("cuda"):
                loss = model(v1.to(device, non_blocking=True), v2.to(device, non_blocking=True))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run, n = run + loss.item(), n + 1
        for g in opt.param_groups:  # SimSiam: the predictor keeps a fixed lr
            if g.get("fix_lr"):
                g["lr"] = args.lr
        sched.step()
        # std of normalised embeddings: ~1/sqrt(dim) healthy, ->0 means collapse
        with torch.no_grad():
            z = F.normalize(model.encode(v1.to(device)), dim=1)
        print(f"epoch {epoch}: loss={run / max(n, 1):.4f} emb_std={z.std(0).mean().item():.4f} "
              f"(healthy ~{1 / 2048 ** 0.5:.4f}) {time.time() - t0:.0f}s "
              f"peak_mem={torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
        torch.save({"body": model.body.state_dict(), "epoch": epoch}, OUT / f"{args.name}.pt")


if __name__ == "__main__":
    main()
