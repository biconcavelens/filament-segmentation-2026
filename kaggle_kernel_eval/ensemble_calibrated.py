"""Cross-architecture ensemble (Mask R-CNN + YOLO11m-seg) with a principled
calibration instead of hand-tuned/grid-searched thresholds.

Three earlier attempts at merging these two detectors all failed despite
confirming genuine, large complementary recall exists (TP up to 569/908 vs
~504-519 for any solo detector): the root cause was that each detector's
raw confidence score is on its own scale (different loss/training dynamics),
so a single shared threshold -- or even hand-tuned per-detector thresholds --
isn't meaningful. This fits an isotonic regression per detector
(raw_score -> P(true positive), monotonic, low-capacity) on the val split,
then pools calibrated candidates from both detectors and paints them with
ONE shared acceptance threshold via the same panoptic-paint used everywhere
else -- the cross-detector IoU-agreement dedup logic becomes unnecessary
once scores are actually comparable, since paint_panoptic's greedy
highest-score-wins-the-pixels logic already resolves overlaps correctly.
"""
import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "scikit-learn"], check=True)

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pycocotools.mask as mu
from PIL import Image
from ultralytics import YOLO
from sklearn.isotonic import IsotonicRegression
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

DATA = Path("/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
IMG_DIR = DATA / "train" / "train_images"
ANN_PATH = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
CKPT = Path("/kaggle/input/datasets/trishanthmellimi/filament-seg-checkpoints")

H, W = 2048, 2048
CROP_SIZE = 256
CONTEXT = 1.8
MIN_CROP = 96
MIN_AREA = 20
REFINER_THRESHOLD = 0.5
TTA_AREA_RATIO_MIN = 0.60
TTA_AREA_RATIO_MAX = 1.70
FLOOR_MASKRCNN = 0.50
FLOOR_YOLO = 0.15
DETECTOR_MASK_THRESHOLD = 0.5
MIN_SIZE, MAX_SIZE = 800, 1333


# ---------- Mask R-CNN ----------

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


def build_model_v3(num_classes=2):
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.ops import MultiScaleRoIAlign
    model = build_model(num_classes)
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)), aspect_ratios=((0.5, 1.0, 2.0),) * 5)
    model.roi_heads.mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=28, sampling_ratio=2)
    model.transform.min_size, model.transform.max_size = (1024,), 1024
    return model


def build_from_checkpoint(state, num_classes=2):
    return build_model_v3(num_classes) if state.get("arch") == "v3" else build_model(num_classes)


@torch.no_grad()
def maskrcnn_raw_candidates(detector, device, img_t, floor):
    out = detector([img_t.to(device)])[0]
    scores = out["scores"].cpu().numpy()
    masks = out["masks"].cpu().numpy()
    cands = []
    for j in range(len(scores)):
        if scores[j] < floor:
            continue
        coarse = (masks[j, 0] > DETECTOR_MASK_THRESHOLD).astype(np.uint8)
        if coarse.sum() == 0:
            continue
        cands.append((float(scores[j]), coarse))
    return cands


@torch.no_grad()
def yolo_raw_candidates(yolo, img_path, floor):
    out = yolo.predict(source=str(img_path), imgsz=1280, conf=floor, verbose=False)[0]
    if out.boxes is None or len(out.boxes) == 0:
        return []
    boxes = out.boxes.xyxy.cpu().numpy()
    scores = out.boxes.conf.cpu().numpy()
    cands = []
    for j in range(len(boxes)):
        x0, y0, x1, y1 = boxes[j]
        x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
        if x1 <= x0 or y1 <= y0:
            continue
        coarse = np.zeros((H, W), dtype=np.uint8)
        coarse[y0:y1, x0:x1] = 1
        cands.append((float(scores[j]), coarse))
    return cands


# ---------- Refiner (exact copies, verified equivalent earlier this session) ----------

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


@torch.no_grad()
def refine_candidate(refiner, device, gray, coarse):
    x0, y0, x1, y1 = square_bounds(coarse)
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


def to_rle(mask):
    return mu.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("utf-8")


def paint_panoptic(candidates, min_area=MIN_AREA):
    candidates = sorted(candidates, key=lambda x: -x[0])
    claimed = np.zeros((H, W), dtype=np.uint8)
    kept = []
    for score, binary in candidates:
        remaining = binary & (1 - claimed)
        area = int(remaining.sum())
        if area < min_area:
            continue
        claimed |= remaining
        kept.append(remaining)
    return kept


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
    val_entries = [i for i in images if i["file_name"] in val_files]
    return val_entries, per_image


def best_iou_against_gt(mask, gt_rles):
    if not gt_rles:
        return 0.0
    pred_rle = {"size": [H, W], "counts": to_rle(mask).encode()}
    iou = mu.iou([pred_rle], gt_rles, [0] * len(gt_rles))
    return float(iou.max())


def pq_against_gt(kept_masks, gt_rles):
    pred = [{"size": [H, W], "counts": to_rle(m).encode()} for m in kept_masks]
    if pred and gt_rles:
        iou = mu.iou(pred, gt_rles, [0] * len(gt_rles))
        best_per_gt = iou.max(axis=0); best_per_pred = iou.max(axis=1)
    else:
        best_per_gt = np.zeros(len(gt_rles)); best_per_pred = np.zeros(len(pred))
    m_tp = int((best_per_gt > 0.5).sum())
    n_fp = len(pred) - m_tp
    n_fn = len(gt_rles) - m_tp
    num = float(best_per_gt[best_per_gt > 0.5].sum())
    den = m_tp + 0.5 * n_fp + 0.5 * n_fn
    return num, den, m_tp


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(CKPT / "maskrcnn_epoch3.pt", map_location=device)
    detA = build_from_checkpoint(state, num_classes=2).to(device)
    detA.load_state_dict(state["model"])
    detA.roi_heads.nms_thresh = 0.30
    detA.eval()

    yolo = YOLO(str(CKPT / "yolo11m_best.pt"))

    rstate = torch.load(CKPT / "refiner_v5_best.pt", map_location=device)
    refiner = RefinerUNet(in_channels=rstate.get("in_channels", 1),
                           out_channels=rstate.get("out_channels", 1)).to(device)
    refiner.load_state_dict(rstate["model"])
    refiner.eval()

    val_entries, per_image = train_val_split()
    print(f"val entries: {len(val_entries)}", flush=True)

    # ---------- cache raw + refined candidates from both detectors ----------
    per_image_cache = []
    scoresA_all, labelsA_all = [], []
    scoresB_all, labelsB_all = [], []

    for i, e in enumerate(val_entries, 1):
        img_path = IMG_DIR / e["file_name"]
        gray = np.array(Image.open(img_path).convert("L"))
        rgb = np.array(Image.open(img_path).convert("RGB"))
        img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        candsA = maskrcnn_raw_candidates(detA, device, img_t, FLOOR_MASKRCNN)
        candsB = yolo_raw_candidates(yolo, img_path, FLOOR_YOLO)

        gt = []
        for a in per_image.get(e["id"], []):
            rles = mu.frPyObjects(a["segmentation"], H, W)
            gt.append({"size": [H, W], "counts": to_rle(mu.decode(mu.merge(rles))).encode()})

        refinedA, refinedB = [], []
        for score, coarse in candsA:
            ref = refine_candidate(refiner, device, gray, coarse)
            label = 1 if best_iou_against_gt(ref, gt) > 0.5 else 0
            refinedA.append((score, ref))
            scoresA_all.append(score); labelsA_all.append(label)
        for score, coarse in candsB:
            ref = refine_candidate(refiner, device, gray, coarse)
            label = 1 if best_iou_against_gt(ref, gt) > 0.5 else 0
            refinedB.append((score, ref))
            scoresB_all.append(score); labelsB_all.append(label)

        per_image_cache.append((refinedA, refinedB, gt))
        if i % 10 == 0:
            print(f"  cached {i}/{len(val_entries)}", flush=True)

    print(f"maskrcnn candidates: {len(scoresA_all)} ({sum(labelsA_all)} TP)", flush=True)
    print(f"yolo candidates: {len(scoresB_all)} ({sum(labelsB_all)} TP)", flush=True)

    # ---------- fit per-detector isotonic calibration ----------
    calA = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calA.fit(scoresA_all, labelsA_all)
    calB = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calB.fit(scoresB_all, labelsB_all)

    print("maskrcnn calibration curve:", flush=True)
    for s in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        print(f"  raw={s} -> calibrated={float(calA.predict([s])[0]):.3f}", flush=True)
    print("yolo calibration curve:", flush=True)
    for s in [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]:
        print(f"  raw={s} -> calibrated={float(calB.predict([s])[0]):.3f}", flush=True)

    # ---------- apply calibration, pool, sweep a single shared threshold ----------
    calibrated_cache = []
    for refinedA, refinedB, gt in per_image_cache:
        pooled = []
        if refinedA:
            calA_scores = calA.predict([s for s, _ in refinedA])
            pooled.extend((float(cs), m) for cs, (_, m) in zip(calA_scores, refinedA))
        if refinedB:
            calB_scores = calB.predict([s for s, _ in refinedB])
            pooled.extend((float(cs), m) for cs, (_, m) in zip(calB_scores, refinedB))
        calibrated_cache.append((pooled, gt))

    results = []
    for thresh in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        pq_num = pq_den = 0.0
        tp_total = 0
        for pooled, gt in calibrated_cache:
            filtered = [(s, m) for s, m in pooled if s >= thresh]
            kept = paint_panoptic(filtered)
            num, den, tp = pq_against_gt(kept, gt)
            pq_num += num; pq_den += den; tp_total += tp
        pq = pq_num / pq_den if pq_den else 0.0
        print(f"threshold={thresh}: PQ={pq:.4f} TP={tp_total}", flush=True)
        results.append((pq, thresh, tp_total))

    results.sort(reverse=True)
    print("\n=== top results (solo baselines: maskrcnn PQ~0.425, "
          "yolo11m PQ=0.4171) ===", flush=True)
    for pq, thresh, tp in results[:5]:
        print(f"PQ={pq:.4f}  threshold={thresh}  TP={tp}")


if __name__ == "__main__":
    main()
