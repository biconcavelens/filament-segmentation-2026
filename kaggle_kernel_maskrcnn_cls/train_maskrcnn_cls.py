"""Apply the same fix that worked for YOLO (real PQ 0.37 -> 0.38: upweight
the classification/confidence loss during training) to Mask R-CNN, our
OTHER detector family (the original ensemble's solo baseline, still at real
PQ 0.37). diag confirmed the same bug exists here too, worse in relative
terms: 231/908 GT filaments (25%) have a well-localized raw box
(bbox-IoU>0.5) but score below our 0.80 deployment threshold, median score
only 0.386 -- less than half the cutoff.

torchvision's Mask R-CNN doesn't expose a `cls` weight like ultralytics;
this reweights `loss_dict["loss_classifier"]` directly in the training
loop, the RoI head's classification loss (the term that produces each
detection's final confidence score).

Saves every epoch's checkpoint (matching the original training's approach,
where epoch 3 of a longer run turned out best -- this small dataset
overfits early) then evaluates ALL of them through the full refiner
pipeline against the held-out val split, sweeping the acceptance threshold
per epoch, to find the best (epoch, threshold) combo -- replicating the
"full checkpoint x threshold sweep" methodology that selected the original
maskrcnn_epoch3.pt.
"""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pycocotools.mask as mu
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

DATA = Path("/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
IMG_DIR = DATA / "train" / "train_images"
ANN_PATH = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
CKPT = Path("/kaggle/input/datasets/trishanthmellimi/filament-seg-checkpoints")
CKPT_OUT = Path("/kaggle/working/checkpoints")
CKPT_OUT.mkdir(parents=True, exist_ok=True)

H, W = 2048, 2048
CROP_SIZE = 256
CONTEXT = 1.8
MIN_CROP = 96
MIN_AREA = 20
REFINER_THRESHOLD = 0.5
TTA_AREA_RATIO_MIN = 0.60
TTA_AREA_RATIO_MAX = 1.70
MIN_SIZE, MAX_SIZE = 800, 1333

CLS_WEIGHT = 3.0  # the lever under test, analogous to YOLO's cls=1.5 (3x its default)
EPOCHS = 6
LR = 0.005
BATCH_SIZE = 2
EVAL_FLOOR = 0.15  # diagnostic found the interesting range is mostly above this
EVAL_EPOCHS_SKIP = 2  # skip evaluating the first N epochs (known to underfit)


# ---------- data ----------

def train_val_split(val_frac=0.1, seed=0):
    with open(ANN_PATH, encoding="utf-8") as f:
        coco = json.load(f)
    per_image = {}
    for a in coco["annotations"]:
        per_image.setdefault(a["image_id"], []).append(a)
    images = coco["images"]
    files = sorted(set(i["file_name"] for i in images))
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * val_frac))
    val_files = set(files[:n_val])
    train_entries = [i for i in images if i["file_name"] not in val_files]
    val_entries = [i for i in images if i["file_name"] in val_files]
    return train_entries, val_entries, per_image


class FilamentDataset(torch.utils.data.Dataset):
    def __init__(self, entries, per_image, augment=False):
        self.entries = entries
        self.per_image = per_image
        self.augment = augment

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img_arr = np.array(Image.open(IMG_DIR / entry["file_name"]).convert("RGB"))

        anns = self.per_image.get(entry["id"], [])
        raw_masks = []
        for a in anns:
            rles = mu.frPyObjects(a["segmentation"], H, W)
            m = mu.decode(mu.merge(rles)).astype(np.uint8)
            if m.sum() == 0:
                continue
            raw_masks.append(m)

        if self.augment:
            if random.random() < 0.5:
                img_arr = np.ascontiguousarray(img_arr[:, ::-1, :])
                raw_masks = [np.ascontiguousarray(m[:, ::-1]) for m in raw_masks]
            if random.random() < 0.5:
                img_arr = np.ascontiguousarray(img_arr[::-1, :, :])
                raw_masks = [np.ascontiguousarray(m[::-1, :]) for m in raw_masks]

        img = torch.from_numpy(img_arr).permute(2, 0, 1).float() / 255.0

        masks, boxes = [], []
        for m in raw_masks:
            ys, xs = np.where(m)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            masks.append(m)

        if masks:
            masks_t = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.ones(len(masks), dtype=torch.int64)
        else:
            masks_t = torch.zeros((0, H, W), dtype=torch.uint8)
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_t, "labels": labels_t, "masks": masks_t,
            "image_id": torch.tensor([idx]),
            "area": (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1]) if len(boxes) else torch.zeros((0,)),
            "iscrowd": torch.zeros((len(masks),), dtype=torch.int64),
        }
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model(num_classes=2):
    model = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
        min_size=MIN_SIZE, max_size=MAX_SIZE,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


# ---------- refiner (verified-equivalent inline copy, same as other kernels) ----------

class ConvBlock(nn.Sequential):
    def __init__(self, in_c, out_c):
        super().__init__(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )


class RefinerUNet(nn.Module):
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


def square_bounds(mask, context=CONTEXT, minimum=MIN_CROP):
    height, width = mask.shape
    ys, xs = np.where(mask)
    cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    side = int(np.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * context))
    side = min(max(side, minimum), min(height, width))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0:
        x1 -= x0; x0 = 0
    if y0 < 0:
        y1 -= y0; y0 = 0
    if x1 > width:
        x0 -= x1 - width; x1 = width
    if y1 > height:
        y0 -= y1 - height; y1 = height
    return x0, y0, x1, y1


@torch.no_grad()
def refine_with_tta(refiner, device, x):
    views = {
        "identity": x, "hflip": x[:, :, ::-1], "vflip": x[:, ::-1, :], "hvflip": x[:, ::-1, ::-1],
    }
    probs = {}
    for name, view in views.items():
        x_t = torch.from_numpy(np.ascontiguousarray(view)).float().unsqueeze(0)
        p = torch.sigmoid(refiner(x_t.to(device)))[0, 0].cpu().numpy()
        if name == "hflip":
            p = np.fliplr(p)
        elif name == "vflip":
            p = np.flipud(p)
        elif name == "hvflip":
            p = np.flipud(np.fliplr(p))
        probs[name] = p
    identity_area = (probs["identity"] > REFINER_THRESHOLD).sum()
    avg_prob = np.mean(list(probs.values()), axis=0)
    if identity_area == 0:
        return avg_prob
    avg_area = (avg_prob > REFINER_THRESHOLD).sum()
    ratio = avg_area / identity_area
    if ratio < TTA_AREA_RATIO_MIN or ratio > TTA_AREA_RATIO_MAX:
        return probs["identity"]
    return avg_prob


def to_rle(mask):
    return mu.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("utf-8")


def paint_panoptic(candidates, min_area=MIN_AREA):
    candidates = sorted(candidates, key=lambda x: -x[0])
    claimed = np.zeros((H, W), dtype=np.uint8)
    kept = []
    for score, binary in candidates:
        remaining = binary & (1 - claimed)
        if int(remaining.sum()) < min_area:
            continue
        claimed |= remaining
        kept.append(remaining)
    return kept


@torch.no_grad()
def raw_candidates(detector, device, img_t, floor):
    out = detector([img_t.to(device)])[0]
    scores = out["scores"].cpu().numpy()
    masks = out["masks"].cpu().numpy()
    cands = []
    for j in range(len(scores)):
        if scores[j] < floor:
            continue
        coarse = (masks[j, 0] > 0.5).astype(np.uint8)
        if coarse.sum() == 0:
            continue
        cands.append((float(scores[j]), coarse))
    return cands


@torch.no_grad()
def refine_candidate(refiner, device, gray, coarse):
    x0, y0, x1, y1 = square_bounds(coarse.astype(bool))
    crop = np.array(Image.fromarray(gray[y0:y1, x0:x1]).resize(
        (CROP_SIZE, CROP_SIZE), Image.BILINEAR)).astype(np.float32) / 255.0
    prob = refine_with_tta(refiner, device, np.stack([crop]))
    side = y1 - y0
    prob_full = np.array(Image.fromarray((prob * 255).astype(np.uint8)).resize(
        (side, side), Image.BILINEAR)).astype(np.float32) / 255.0
    refined_crop = (prob_full > 0.5).astype(np.uint8)
    full_mask = np.zeros((H, W), dtype=np.uint8)
    full_mask[y0:y1, x0:x1] = refined_crop
    return full_mask


def build_cache_for_detector(detector, refiner, device, val_entries, per_image, floor=0.05):
    cache = []
    for i, e in enumerate(val_entries, 1):
        img_path = IMG_DIR / e["file_name"]
        gray = np.array(Image.open(img_path).convert("L"))
        rgb = np.array(Image.open(img_path).convert("RGB"))
        img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        raw = raw_candidates(detector, device, img_t, floor)
        refined = [(s, refine_candidate(refiner, device, gray, c)) for s, c in raw]
        gt = []
        for a in per_image.get(e["id"], []):
            rles = mu.frPyObjects(a["segmentation"], H, W)
            gt.append(to_rle(mu.decode(mu.merge(rles))))
        cache.append((refined, gt))
    return cache


def pq_for(cache, thresh):
    pq_num = pq_den = 0.0
    tp = 0
    for refined, gt in cache:
        filtered = [(s, m) for s, m in refined if s >= thresh]
        kept = paint_panoptic(filtered)
        pred = [{"size": [H, W], "counts": to_rle(m).encode()} for m in kept]
        gt_rle = [{"size": [H, W], "counts": r.encode()} for r in gt]
        if pred and gt_rle:
            iou = mu.iou(pred, gt_rle, [0] * len(gt_rle))
            best_per_gt = iou.max(axis=0)
        else:
            best_per_gt = np.zeros(len(gt_rle))
        m_tp = int((best_per_gt > 0.5).sum())
        n_fp = len(pred) - m_tp
        n_fn = len(gt_rle) - m_tp
        pq_num += float(best_per_gt[best_per_gt > 0.5].sum())
        pq_den += m_tp + 0.5 * n_fp + 0.5 * n_fn
        tp += m_tp
    return (pq_num / pq_den if pq_den else 0.0), tp


def main():
    device = torch.device("cuda")
    print(f"device={device} CLS_WEIGHT={CLS_WEIGHT} EPOCHS={EPOCHS}", flush=True)

    train_entries, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)
    train_ds = FilamentDataset(train_entries, per_image, augment=True)
    val_ds_for_loss = FilamentDataset(val_entries, per_image)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)

    model = build_model(num_classes=2).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    milestones = sorted({max(1, round(EPOCHS * 0.7)), max(2, round(EPOCHS * 0.9))})
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    saved_epochs = []
    for epoch in range(EPOCHS):
        model.train()
        t0 = time.time()
        running = 0.0
        for i, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=True):
                loss_dict = model(images, targets)
                loss = (loss_dict["loss_box_reg"] + loss_dict["loss_mask"] +
                        loss_dict["loss_objectness"] + loss_dict["loss_rpn_box_reg"] +
                        CLS_WEIGHT * loss_dict["loss_classifier"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()

            if (i + 1) % 100 == 0:
                print(f"  epoch {epoch} step {i+1}/{len(train_loader)} loss={running/(i+1):.3f}", flush=True)

        scheduler.step()
        dt = time.time() - t0
        print(f"epoch {epoch}: train_loss={running/len(train_loader):.3f} ({dt:.0f}s)", flush=True)

        ckpt = CKPT_OUT / f"maskrcnn_cls_epoch{epoch}.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch, "arch": None}, ckpt)
        saved_epochs.append(epoch)
        print(f"saved {ckpt}", flush=True)

    print("\ntraining done, selecting best (epoch, threshold) via full refiner pipeline...", flush=True)

    rstate = torch.load(CKPT / "refiner_v5_best.pt", map_location=device)
    refiner = RefinerUNet(in_channels=rstate.get("in_channels", 1),
                           out_channels=rstate.get("out_channels", 1)).to(device)
    refiner.load_state_dict(rstate["model"])
    refiner.eval()

    results = []
    for epoch in saved_epochs:
        if epoch < EVAL_EPOCHS_SKIP:
            continue
        ckpt_path = CKPT_OUT / f"maskrcnn_cls_epoch{epoch}.pt"
        state = torch.load(ckpt_path, map_location=device)
        det = build_model(num_classes=2).to(device)
        det.load_state_dict(state["model"])
        det.roi_heads.nms_thresh = 0.30
        det.eval()

        cache = build_cache_for_detector(det, refiner, device, val_entries, per_image, floor=EVAL_FLOOR)
        for thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            pq, tp = pq_for(cache, thresh)
            print(f"epoch={epoch} thresh={thresh}: PQ={pq:.4f} TP={tp}", flush=True)
            results.append((pq, epoch, thresh, tp))

    results.sort(reverse=True)
    print("\n=== top 10 (original maskrcnn_epoch3.pt baseline: local PQ ~0.351 solo, "
          "~0.425 with refiner per README) ===")
    for pq, epoch, thresh, tp in results[:10]:
        print(f"PQ={pq:.4f}  epoch={epoch}  thresh={thresh}  TP={tp}")

    best_pq, best_epoch, best_thresh, best_tp = results[0]
    best_ckpt = CKPT_OUT / f"maskrcnn_cls_epoch{best_epoch}.pt"
    out = Path("/kaggle/working/maskrcnn_cls_best.pt")
    import shutil
    shutil.copy(best_ckpt, out)
    print(f"\nbest: epoch={best_epoch} thresh={best_thresh} PQ={best_pq:.4f} -> copied to {out}", flush=True)


if __name__ == "__main__":
    main()
