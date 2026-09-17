"""Cache YOLO's raw detections + refined masks ONCE at a low confidence
floor, then sweep the confidence threshold cheaply against the cache.
Much lighter than the dual-detector ensemble sweep (single detector, single
refine pass per candidate) -- the ensemble sweep crashed twice from system
resource strain, so keep this one lean and check system health first.
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
from predict_refined import refine_with_tta

FLOOR = 0.05


@torch.no_grad()
def refine_candidate(refiner, device, gray_img, coarse):
    x0, y0, x1, y1 = square_bounds(coarse)
    crop = np.array(Image.fromarray(gray_img[y0:y1, x0:x1]).resize(
        (CROP_SIZE, CROP_SIZE), Image.BILINEAR)).astype(np.float32) / 255.0
    prob = refine_with_tta(refiner, device, np.stack([crop]))
    side = y1 - y0
    prob_full = np.array(Image.fromarray((prob * 255).astype(np.uint8)).resize(
        (side, side), Image.BILINEAR)).astype(np.float32) / 255.0
    refined_crop = (prob_full > 0.5).astype(np.uint8)
    full_mask = np.zeros((H, W), dtype=np.uint8)
    full_mask[y0:y1, x0:x1] = refined_crop
    return full_mask


def paint_panoptic(candidates, min_area=20):
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


def main():
    device = torch.device("cuda")
    yolo = YOLO("runs/segment/yolo_runs/filament/weights/best.pt")

    rstate = torch.load("checkpoints/refiner_v5_best.pt", map_location=device)
    refiner = RefinerUNet(in_channels=rstate.get("in_channels", 1),
                           out_channels=rstate.get("out_channels", 1)).to(device)
    refiner.load_state_dict(rstate["model"]); refiner.eval()

    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)

    cache = []
    for i, e in enumerate(val_entries, 1):
        img_path = IMG_DIR / e["file_name"]
        gray = np.array(Image.open(img_path).convert("L"))
        out = yolo.predict(source=str(img_path), imgsz=1280, conf=FLOOR, verbose=False)[0]

        cands = []
        if out.boxes is not None and len(out.boxes) > 0:
            boxes = out.boxes.xyxy.cpu().numpy()
            scores = out.boxes.conf.cpu().numpy()
            for j in range(len(boxes)):
                x0, y0, x1, y1 = boxes[j]
                x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
                if x1 <= x0 or y1 <= y0:
                    continue
                coarse = np.zeros((H, W), dtype=np.uint8)
                coarse[y0:y1, x0:x1] = 1
                ref = refine_candidate(refiner, device, gray, coarse)
                cands.append((float(scores[j]), ref))

        gt = []
        for a in per_image.get(e["id"], []):
            rles = mu.frPyObjects(a["segmentation"], H, W)
            gt.append(to_rle(mu.decode(mu.merge(rles))))

        cache.append((cands, gt))
        if i % 20 == 0:
            print(f"  cached {i}/{len(val_entries)}", flush=True)

    print("cache built, sweeping confidence thresholds...\n")

    def pq_for(conf_thresh):
        pq_num = pq_den = 0.0
        tp = fp_spur = n_pred_total = 0
        for cands, gt in cache:
            filtered = [(s, m) for s, m in cands if s >= conf_thresh]
            kept = paint_panoptic(filtered)
            pred = [{"size": [H, W], "counts": to_rle(m).encode()} for m in kept]
            gt_rle = [{"size": [H, W], "counts": r.encode()} for r in gt]
            if pred and gt_rle:
                iou = mu.iou(pred, gt_rle, [0] * len(gt_rle))
                best_per_gt = iou.max(axis=0); best_per_pred = iou.max(axis=1)
            else:
                best_per_gt = np.zeros(len(gt_rle)); best_per_pred = np.zeros(len(pred))
            m_tp = int((best_per_gt > 0.5).sum())
            n_fp = len(pred) - m_tp
            n_fn = len(gt_rle) - m_tp
            pq_num += float(best_per_gt[best_per_gt > 0.5].sum())
            pq_den += m_tp + 0.5 * n_fp + 0.5 * n_fn
            tp += m_tp
            fp_spur += int((best_per_pred <= 0.2).sum())
            n_pred_total += len(pred)
        pq = pq_num / pq_den if pq_den else 0.0
        return pq, tp, fp_spur, n_pred_total

    results = []
    for conf in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
        pq, tp, fp_spur, n_pred = pq_for(conf)
        print(f"conf={conf}: PQ={pq:.4f} TP={tp} FP_spur={fp_spur} n_pred={n_pred}", flush=True)
        results.append((pq, conf))

    results.sort(reverse=True)
    print("\n=== top 5 ===")
    for pq, conf in results[:5]:
        print(f"PQ={pq:.4f}  conf={conf}")


if __name__ == "__main__":
    main()
