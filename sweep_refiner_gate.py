"""Diagnostic finding (diag_missed_filaments.py + manual check): several
large, obvious filaments are missed entirely not because YOLO fails to
localize them (raw box-IoU vs GT was 0.7-0.87) but because YOLO's own
confidence score badly underrates the correct box (score as low as 0.05,
7x below our 0.35 deployment threshold) -- yet feeding that exact box to
the refiner produces a mask at IoU=0.75 vs GT with 93% mean refiner
confidence. The refiner's own output confidence looks like a much better
acceptance signal than YOLO's raw box score for these cases.

Cached single-axis sweep, same validated methodology as sweep_yolo_solo.py:
lower YOLO's floor to catch these low-confidence-but-well-localized boxes,
refine ALL of them once, then sweep the acceptance threshold on the
REFINER's own mean-probability-inside-mask instead of YOLO's raw score.
"""
import numpy as np
import torch
import pycocotools.mask as mu
from PIL import Image
from ultralytics import YOLO

from dataset import train_val_split, IMG_DIR, H, W
from train_refiner import RefinerUNet
from crop_dataset import square_bounds, CROP_SIZE
from predict_trained import to_rle

YOLO_CKPT = "kaggle_kernel_train_m/output/yolo11m_best.pt"
REFINER_CKPT = "checkpoints/refiner_v5_best.pt"
YOLO_FLOOR = 0.02  # much lower than the deployed 0.35 -- catch the missed boxes
MIN_AREA = 20
REFINER_THRESHOLD = 0.5
TTA_AREA_RATIO_MIN = 0.60
TTA_AREA_RATIO_MAX = 1.70


@torch.no_grad()
def refine_with_tta_and_confidence(refiner, device, x):
    """Same as predict_refined.refine_with_tta, but also returns the mean
    probability inside the final binary mask as a refiner-confidence score."""
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
        final = avg_prob
    else:
        avg_area = (avg_prob > REFINER_THRESHOLD).sum()
        ratio = avg_area / identity_area
        final = probs["identity"] if (ratio < TTA_AREA_RATIO_MIN or ratio > TTA_AREA_RATIO_MAX) else avg_prob

    mask = final > REFINER_THRESHOLD
    refiner_conf = float(final[mask].mean()) if mask.sum() > 0 else 0.0
    return final, refiner_conf


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


def main():
    device = torch.device("cuda")
    yolo = YOLO(YOLO_CKPT)
    rstate = torch.load(REFINER_CKPT, map_location=device)
    refiner = RefinerUNet(in_channels=rstate.get("in_channels", 1),
                           out_channels=rstate.get("out_channels", 1)).to(device)
    refiner.load_state_dict(rstate["model"])
    refiner.eval()

    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)

    cache = []
    for i, e in enumerate(val_entries, 1):
        img_path = IMG_DIR / e["file_name"]
        gray = np.array(Image.open(img_path).convert("L"))
        out = yolo.predict(source=str(img_path), imgsz=1280, conf=YOLO_FLOOR, verbose=False)[0]

        cands = []
        if out.boxes is not None and len(out.boxes) > 0:
            boxes = out.boxes.xyxy.cpu().numpy()
            yolo_scores = out.boxes.conf.cpu().numpy()
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
                prob, refiner_conf = refine_with_tta_and_confidence(refiner, device, np.stack([crop]))
                side = cy1 - cy0
                prob_full = np.array(Image.fromarray((prob * 255).astype(np.uint8)).resize(
                    (side, side), Image.BILINEAR)).astype(np.float32) / 255.0
                refined_crop = (prob_full > 0.5).astype(np.uint8)
                full_mask = np.zeros((H, W), dtype=np.uint8)
                full_mask[cy0:cy1, cx0:cx1] = refined_crop
                if full_mask.sum() == 0:
                    continue
                cands.append((float(yolo_scores[j]), refiner_conf, full_mask))

        gt = []
        for a in per_image.get(e["id"], []):
            rles = mu.frPyObjects(a["segmentation"], H, W)
            gt.append(to_rle(mu.decode(mu.merge(rles))))

        cache.append((cands, gt))
        if i % 10 == 0:
            print(f"  cached {i}/{len(val_entries)} (avg {sum(len(c[0]) for c in cache)/i:.1f} "
                  f"candidates/image at floor={YOLO_FLOOR})", flush=True)

    print("cache built, sweeping acceptance signal...\n", flush=True)

    def pq_for(score_fn, thresh):
        pq_num = pq_den = 0.0
        tp = 0
        for cands, gt in cache:
            filtered = [(score_fn(yc, rc), m) for yc, rc, m in cands if score_fn(yc, rc) >= thresh]
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

    print("=== baseline: YOLO raw confidence gate (current deployed approach) ===")
    results_yolo = []
    for thresh in [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        pq, tp = pq_for(lambda y, r: y, thresh)
        print(f"yolo_conf>={thresh}: PQ={pq:.4f} TP={tp}", flush=True)
        results_yolo.append((pq, thresh, tp))

    print("\n=== new: REFINER confidence gate (ignore YOLO's score entirely) ===")
    results_refiner = []
    for thresh in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
        pq, tp = pq_for(lambda y, r: r, thresh)
        print(f"refiner_conf>={thresh}: PQ={pq:.4f} TP={tp}", flush=True)
        results_refiner.append((pq, thresh, tp))

    print("\n=== combined: refiner confidence gate + tiny YOLO floor (>=0.05, filters pure junk) ===")
    results_combo = []
    for thresh in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
        pq, tp = pq_for(lambda y, r: r if y >= 0.05 else 0.0, thresh)
        print(f"yolo>=0.05 & refiner_conf>={thresh}: PQ={pq:.4f} TP={tp}", flush=True)
        results_combo.append((pq, thresh, tp))

    print("\n=== top 5 overall ===")
    all_results = ([(pq, f"yolo_conf>={t}", tp) for pq, t, tp in results_yolo] +
                    [(pq, f"refiner_conf>={t}", tp) for pq, t, tp in results_refiner] +
                    [(pq, f"combo refiner_conf>={t}", tp) for pq, t, tp in results_combo])
    all_results.sort(reverse=True)
    for pq, label, tp in all_results[:5]:
        print(f"PQ={pq:.4f}  {label}  TP={tp}")


if __name__ == "__main__":
    main()
