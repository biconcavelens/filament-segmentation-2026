"""Characterize the ~45% of GT filaments our best pipeline misses entirely
(no prediction with IoU>0.5): size, elongation, and contrast, compared
against the filaments we DO catch. Also saves crop images of the worst
misses for visual inspection. Goal: find a specific, addressable pattern
(e.g. "we miss anything under N px wide") instead of another blind
architecture swap.
"""
import numpy as np
import torch
import pycocotools.mask as mu
import cv2
from PIL import Image, ImageDraw
from pathlib import Path
from ultralytics import YOLO

from dataset import train_val_split, IMG_DIR, H, W
from train_refiner import RefinerUNet
from crop_dataset import square_bounds, CROP_SIZE
from predict_trained import to_rle
from predict_refined import refine_with_tta

YOLO_CKPT = "kaggle_kernel_train_m/output/yolo11m_best.pt"
REFINER_CKPT = "checkpoints/refiner_v5_best.pt"
YOLO_CONF = 0.35
MIN_AREA = 20
OUT_DIR = Path("scratch_missed_filaments")
OUT_DIR.mkdir(exist_ok=True)


@torch.no_grad()
def predict_one(yolo, refiner, device, gray, img_path):
    out = yolo.predict(source=str(img_path), imgsz=1280, conf=YOLO_CONF, verbose=False)[0]
    if out.boxes is None or len(out.boxes) == 0:
        return []
    boxes = out.boxes.xyxy.cpu().numpy()
    scores = out.boxes.conf.cpu().numpy()
    candidates = []
    for j in range(len(boxes)):
        x0, y0, x1, y1 = boxes[j]
        x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
        if x1 <= x0 or y1 <= y0:
            continue
        coarse = np.zeros((H, W), dtype=np.uint8)
        coarse[y0:y1, x0:x1] = 1
        cx0, cy0, cx1, cy1 = square_bounds(coarse.astype(bool))
        crop = np.array(Image.fromarray(gray[cy0:cy1, cx0:cx1]).resize(
            (CROP_SIZE, CROP_SIZE), Image.BILINEAR)).astype(np.float32) / 255.0
        prob = refine_with_tta(refiner, device, np.stack([crop]))
        side = cy1 - cy0
        prob_full = np.array(Image.fromarray((prob * 255).astype(np.uint8)).resize(
            (side, side), Image.BILINEAR)).astype(np.float32) / 255.0
        refined_crop = (prob_full > 0.5).astype(np.uint8)
        full_mask = np.zeros((H, W), dtype=np.uint8)
        full_mask[cy0:cy1, cx0:cx1] = refined_crop
        if full_mask.sum() == 0:
            continue
        candidates.append((float(scores[j]), full_mask))

    candidates.sort(key=lambda x: -x[0])
    claimed = np.zeros((H, W), dtype=np.uint8)
    kept = []
    for score, binary in candidates:
        remaining = binary & (1 - claimed)
        if int(remaining.sum()) < MIN_AREA:
            continue
        claimed |= remaining
        kept.append(remaining)
    return kept


def mask_stats(mask, gray):
    ys, xs = np.where(mask)
    area = len(ys)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    h, w = y1 - y0, x1 - x0
    long_side, short_side = max(h, w), max(1, min(h, w))
    elongation = long_side / short_side

    # thickness proxy: area / long_side (a thin filament has small thickness
    # even if its bounding box is large)
    thickness = area / max(1, long_side)

    inside_mean = gray[mask.astype(bool)].mean()
    # local background ring: dilate the bbox by 15px, exclude the mask itself
    pad = 15
    by0, by1 = max(0, y0 - pad), min(H, y1 + pad)
    bx0, bx1 = max(0, x0 - pad), min(W, x1 + pad)
    region = gray[by0:by1, bx0:bx1].astype(np.float32)
    region_mask = mask[by0:by1, bx0:bx1].astype(bool)
    bg_pixels = region[~region_mask]
    bg_mean = bg_pixels.mean() if bg_pixels.size else inside_mean
    contrast = bg_mean - inside_mean  # filaments are dark -> positive = darker than surroundings

    return dict(area=area, long_side=long_side, short_side=short_side,
                elongation=elongation, thickness=thickness,
                inside_mean=inside_mean, bg_mean=bg_mean, contrast=contrast)


def save_crop(img_gray, gt_mask, pred_masks, out_path, pad=60):
    ys, xs = np.where(gt_mask)
    y0, y1, x0, x1 = max(0, ys.min() - pad), min(H, ys.max() + pad), \
                      max(0, xs.min() - pad), min(W, xs.max() + pad)
    crop = img_gray[y0:y1, x0:x1]
    rgb = np.stack([crop] * 3, axis=-1).astype(np.uint8)
    im = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(im)

    gt_crop = gt_mask[y0:y1, x0:x1].astype(np.uint8)
    contours, _ = cv2.findContours(gt_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        pts = [tuple(p[0]) for p in c]
        if len(pts) > 1:
            draw.line(pts + [pts[0]], fill=(0, 255, 0), width=2)  # GT in green

    for pm in pred_masks:
        pm_crop = pm[y0:y1, x0:x1].astype(np.uint8)
        if pm_crop.sum() == 0:
            continue
        contours, _ = cv2.findContours(pm_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            pts = [tuple(p[0]) for p in c]
            if len(pts) > 1:
                draw.line(pts + [pts[0]], fill=(255, 0, 0), width=2)  # our preds in red

    im.save(out_path)


def main():
    device = torch.device("cuda")
    yolo = YOLO(YOLO_CKPT)
    rstate = torch.load(REFINER_CKPT, map_location=device)
    refiner = RefinerUNet(in_channels=rstate.get("in_channels", 1),
                           out_channels=rstate.get("out_channels", 1)).to(device)
    refiner.load_state_dict(rstate["model"])
    refiner.eval()

    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)

    missed_stats = []
    caught_stats = []
    missed_examples = []  # (file, gt_mask, pred_masks, area) for crop-saving

    for i, e in enumerate(val_entries, 1):
        img_path = IMG_DIR / e["file_name"]
        gray = np.array(Image.open(img_path).convert("L"))
        kept = predict_one(yolo, refiner, device, gray, img_path)
        pred = [{"size": [H, W], "counts": to_rle(m).encode()} for m in kept]

        gt_masks = []
        for a in per_image.get(e["id"], []):
            rles = mu.frPyObjects(a["segmentation"], H, W)
            gt_masks.append(mu.decode(mu.merge(rles)))
        gt_rle = [{"size": [H, W], "counts": to_rle(m).encode()} for m in gt_masks]

        if pred and gt_rle:
            iou = mu.iou(pred, gt_rle, [0] * len(gt_rle))
            best_per_gt = iou.max(axis=0)
        else:
            best_per_gt = np.zeros(len(gt_rle))

        for gi, gt_mask in enumerate(gt_masks):
            stats = mask_stats(gt_mask, gray)
            if best_per_gt[gi] <= 0.2:  # total miss (matches session's established near-miss/total-miss split)
                missed_stats.append(stats)
                missed_examples.append((e["file_name"], gt_mask, kept, stats["area"]))
            elif best_per_gt[gi] > 0.5:
                caught_stats.append(stats)

        if i % 20 == 0:
            print(f"  {i}/{len(val_entries)}", flush=True)

    def summarize(name, stats_list):
        if not stats_list:
            print(f"{name}: none")
            return
        areas = [s["area"] for s in stats_list]
        elong = [s["elongation"] for s in stats_list]
        thick = [s["thickness"] for s in stats_list]
        contrast = [s["contrast"] for s in stats_list]
        print(f"{name} (n={len(stats_list)}):")
        print(f"  area:       median={np.median(areas):.0f}  mean={np.mean(areas):.0f}  "
              f"p25={np.percentile(areas,25):.0f}  p75={np.percentile(areas,75):.0f}")
        print(f"  elongation: median={np.median(elong):.2f}  mean={np.mean(elong):.2f}")
        print(f"  thickness:  median={np.median(thick):.2f}px  mean={np.mean(thick):.2f}px")
        print(f"  contrast:   median={np.median(contrast):.1f}  mean={np.mean(contrast):.1f} "
              f"(gray levels, higher = darker than surroundings = easier)")

    print(f"\n=== total-miss GT: {len(missed_stats)}, caught GT (TP): {len(caught_stats)} ===\n")
    summarize("MISSED (total-miss, IoU<=0.2)", missed_stats)
    print()
    summarize("CAUGHT (TP, IoU>0.5)", caught_stats)

    # save crops of the biggest misses (most surprising: large area but still missed)
    missed_examples.sort(key=lambda x: -x[3])
    print(f"\nsaving crops of the 10 LARGEST missed filaments to {OUT_DIR}/ ...")
    for k, (fname, gt_mask, preds, area) in enumerate(missed_examples[:10], 1):
        gray = np.array(Image.open(IMG_DIR / fname).convert("L"))
        out_path = OUT_DIR / f"{k:02d}_area{area}_{fname.replace('.jpeg','')}.png"
        save_crop(gray, gt_mask, preds, out_path)
        print(f"  {out_path}")

    print(f"\nsaving crops of 10 SMALLEST missed filaments (typical case) ...")
    for k, (fname, gt_mask, preds, area) in enumerate(sorted(missed_examples, key=lambda x: x[3])[:10], 11):
        gray = np.array(Image.open(IMG_DIR / fname).convert("L"))
        out_path = OUT_DIR / f"{k:02d}_area{area}_{fname.replace('.jpeg','')}.png"
        save_crop(gray, gt_mask, preds, out_path)
        print(f"  {out_path}")


if __name__ == "__main__":
    main()
