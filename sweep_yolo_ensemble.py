"""Cache refined candidates from BOTH the Mask R-CNN and YOLO detectors
once (the expensive part), then cheaply sweep the cross-architecture merge
thresholds against the cache. Three single-shot full evaluations already
confirmed genuine, large complementary recall exists (TP swings from 350 to
569 depending on thresholds) -- a real structural signal worth a proper
search, unlike the earlier same-architecture threshold tuning that was
searching pure noise on a already-converged local optimum.
"""
import numpy as np
import torch
import pycocotools.mask as mu
from PIL import Image
from ultralytics import YOLO

from dataset import train_val_split, IMG_DIR, H, W
from train import build_from_checkpoint
from train_refiner import RefinerUNet
from crop_dataset import square_bounds, CROP_SIZE
from predict_trained import to_rle
from predict_refined import refine_with_tta, DETECTOR_MASK_THRESHOLD
from ensemble_diag import raw_candidates as maskrcnn_raw_candidates

FLOOR_MASKRCNN = 0.55
FLOOR_YOLO = 0.25


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


def dedup(candsA, candsB, dedup_iou, agree_score, maskrcnn_unique, yolo_unique):
    usedB = [False] * len(candsB)
    kept = []
    for scoreA, coarseA, refA in candsA:
        best_j, best_iou = -1, 0.0
        for j, (scoreB, coarseB, refB) in enumerate(candsB):
            if usedB[j]:
                continue
            inter = np.logical_and(coarseA, coarseB).sum()
            if inter == 0:
                continue
            union = np.logical_or(coarseA, coarseB).sum()
            iou = inter / union
            if iou > dedup_iou and iou > best_iou:
                best_j, best_iou = j, iou
        if best_j >= 0:
            usedB[best_j] = True
            scoreB, coarseB, refB = candsB[best_j]
            if max(scoreA, scoreB) >= agree_score:
                kept.append((max(scoreA, scoreB), refA if scoreA >= scoreB else refB))
        else:
            if scoreA >= maskrcnn_unique:
                kept.append((scoreA, refA))
    for j, (scoreB, coarseB, refB) in enumerate(candsB):
        if not usedB[j] and scoreB >= yolo_unique:
            kept.append((scoreB, refB))
    return kept


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
    state = torch.load("checkpoints/maskrcnn_epoch3.pt", map_location=device)
    detA = build_from_checkpoint(state, num_classes=2).to(device)
    detA.load_state_dict(state["model"]); detA.roi_heads.nms_thresh = 0.30; detA.eval()

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
        rgb = np.array(Image.open(img_path).convert("RGB"))
        img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        candsA = maskrcnn_raw_candidates(detA, device, img_t, floor=FLOOR_MASKRCNN)
        candsB = yolo_raw_candidates(yolo, img_path, floor=FLOOR_YOLO)
        candsA = [(s, c, refine_candidate(refiner, device, gray, c)) for s, c in candsA]
        candsB = [(s, c, refine_candidate(refiner, device, gray, c)) for s, c in candsB]

        gt = []
        for a in per_image.get(e["id"], []):
            rles = mu.frPyObjects(a["segmentation"], H, W)
            gt.append(to_rle(mu.decode(mu.merge(rles))))

        cache.append((candsA, candsB, gt))
        if i % 10 == 0:
            print(f"  cached {i}/{len(val_entries)}", flush=True)

    print("cache built, sweeping thresholds...\n")

    def pq_for(dedup_iou, agree_score, maskrcnn_unique, yolo_unique):
        pq_num = pq_den = 0.0
        tp = fp_spur = n_pred_total = 0
        for candsA, candsB, gt in cache:
            merged = dedup(candsA, candsB, dedup_iou, agree_score, maskrcnn_unique, yolo_unique)
            kept = paint_panoptic(merged)
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
    for maskrcnn_unique in [0.80, 0.85, 0.90, 0.95, 0.97]:
        for yolo_unique in [0.35, 0.45, 0.55, 0.65, 0.75]:
            pq, tp, fp_spur, n_pred = pq_for(0.5, 0.5, maskrcnn_unique, yolo_unique)
            print(f"mrcnn_unique={maskrcnn_unique} yolo_unique={yolo_unique}: "
                  f"PQ={pq:.4f} TP={tp} FP_spur={fp_spur} n_pred={n_pred}", flush=True)
            results.append((pq, maskrcnn_unique, yolo_unique))

    results.sort(reverse=True)
    print("\n=== top 5 ===")
    for pq, mu_, yu in results[:5]:
        print(f"PQ={pq:.4f}  mrcnn_unique={mu_} yolo_unique={yu}")


if __name__ == "__main__":
    main()
