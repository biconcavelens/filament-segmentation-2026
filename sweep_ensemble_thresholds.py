"""Cache refined candidates from both detectors ONCE (the expensive part:
detector forward pass + crop-refine + TTA), then cheaply sweep the
agreement-weighted merge thresholds (AGREE_SCORE, UNIQUE_SCORE, DEDUP_IOU)
and the detector floor score against the cache -- no GPU re-inference
needed per combo. These thresholds were last tuned for an older detector
pairing (epoch3 + cp_epoch7) and never retuned for the current best trio
(full-data detA + tile detB + spine-supervised refiner).
"""
import numpy as np
import torch
import pycocotools.mask as mu
from PIL import Image

from dataset import train_val_split, IMG_DIR, H, W
from train import build_from_checkpoint
from train_refiner import RefinerUNet
from crop_dataset import square_bounds, CROP_SIZE
from predict_trained import to_rle
from predict_refined import refine_with_tta, DETECTOR_MASK_THRESHOLD

FLOOR_SCORE = 0.35  # cache everything down to this floor; sweep can only threshold above it


@torch.no_grad()
def raw_candidates(detector, device, img_t, floor=FLOOR_SCORE):
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


def dedup(candsA, candsB, dedup_iou, agree_score, unique_score):
    """candsA/candsB: list of (score, coarse_mask, refined_mask)."""
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
            score = max(scoreA, scoreB)
            if score >= agree_score:
                kept.append((score, refA if scoreA >= scoreB else refB))
        else:
            if scoreA >= unique_score:
                kept.append((scoreA, refA))
    for j, (scoreB, coarseB, refB) in enumerate(candsB):
        if not usedB[j] and scoreB >= unique_score:
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
    stateA = torch.load("checkpoints/maskrcnn_full_epoch7.pt", map_location=device)
    detA = build_from_checkpoint(stateA, num_classes=2).to(device)
    detA.load_state_dict(stateA["model"]); detA.roi_heads.nms_thresh = 0.30; detA.eval()

    stateB = torch.load("checkpoints/maskrcnn_tile_epoch7.pt", map_location=device)
    detB = build_from_checkpoint(stateB, num_classes=2).to(device)
    detB.load_state_dict(stateB["model"]); detB.roi_heads.nms_thresh = 0.30; detB.eval()

    rstate = torch.load("checkpoints/refiner_v5_full_best.pt", map_location=device)
    refiner = RefinerUNet(in_channels=rstate.get("in_channels", 1),
                           out_channels=rstate.get("out_channels", 1)).to(device)
    refiner.load_state_dict(rstate["model"]); refiner.eval()

    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)

    cache = []
    for i, e in enumerate(val_entries, 1):
        gray = np.array(Image.open(IMG_DIR / e["file_name"]).convert("L"))
        rgb = np.array(Image.open(IMG_DIR / e["file_name"]).convert("RGB"))
        img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

        candsA = raw_candidates(detA, device, img_t)
        candsB = raw_candidates(detB, device, img_t)
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

    def pq_for(dedup_iou, agree_score, unique_score, min_area=20):
        pq_num = pq_den = 0.0
        for candsA, candsB, gt in cache:
            merged = dedup(candsA, candsB, dedup_iou, agree_score, unique_score)
            kept = paint_panoptic(merged, min_area)
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
        return pq_num / pq_den if pq_den else 0.0

    baseline = pq_for(0.5, 0.80, 0.95)
    print(f"baseline (dedup=0.5 agree=0.80 unique=0.95): PQ={baseline:.4f}\n")

    results = []
    for agree in [0.70, 0.75, 0.80, 0.85]:
        for unique in [0.85, 0.90, 0.95, 0.97]:
            pq = pq_for(0.5, agree, unique)
            print(f"agree={agree} unique={unique}: PQ={pq:.4f}", flush=True)
            results.append((pq, agree, unique))

    results.sort(reverse=True)
    print("\n=== top 5 ===")
    for pq, agree, unique in results[:5]:
        print(f"PQ={pq:.4f}  agree={agree} unique={unique}")


if __name__ == "__main__":
    main()
