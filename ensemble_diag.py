"""Two-detector candidate-union ensemble: merge raw detections from two
independently trained Mask R-CNN checkpoints before crop-refine + painting.
Different training runs can miss different filaments, so union (with IoU
dedup) might recover total-misses that neither detector alone catches --
unlike the earlier Mask R-CNN + U-Net ensemble (too architecturally
different to complement), these are the same architecture/task so their
candidate boxes are directly comparable.
"""
import numpy as np
import torch
import pycocotools.mask as mu
from PIL import Image

from dataset import train_val_split, IMG_DIR, H, W
from train import build_from_checkpoint
from train_refiner import RefinerUNet
from crop_dataset import square_bounds, CROP_SIZE
from predict_trained import paint_panoptic, to_rle
from predict_refined import refine_with_tta, DETECTOR_SCORE_THRESHOLD, DETECTOR_MASK_THRESHOLD, MIN_AREA

DEDUP_IOU = 0.5      # two detectors' candidates for the same filament get merged, higher score wins
FLOOR_SCORE = 0.50    # gather candidates loosely so cross-detector agreement is visible...
AGREE_SCORE = 0.80    # ...but only KEEP an agreed-upon pair at the normal bar
UNIQUE_SCORE = 0.95   # a candidate only ONE detector found needs to be much more confident
# Both a 16-way (agree, unique) grid AND the floor change alone (0.50->0.35,
# tested in isolation) looked better locally (0.407->0.424-0.430) but BOTH
# scored worse on the real leaderboard (0.37->0.36) -- confirmed twice, not a
# one-off. Loosening the candidate floor does not generalize past this val
# set; do not touch these thresholds again without new independent evidence.


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


def dedup(candsA, candsB):
    """Merge A+B by IoU. A pair both detectors found needs only AGREE_SCORE;
    a candidate unique to one detector needs the stricter UNIQUE_SCORE."""
    usedB = [False] * len(candsB)
    kept = []
    for scoreA, maskA in candsA:
        best_j, best_iou = -1, 0.0
        for j, (scoreB, maskB) in enumerate(candsB):
            if usedB[j]:
                continue
            inter = np.logical_and(maskA, maskB).sum()
            if inter == 0:
                continue
            union = np.logical_or(maskA, maskB).sum()
            iou = inter / union
            if iou > DEDUP_IOU and iou > best_iou:
                best_j, best_iou = j, iou
        if best_j >= 0:
            usedB[best_j] = True
            scoreB, maskB = candsB[best_j]
            score = max(scoreA, scoreB)
            if score >= AGREE_SCORE:
                kept.append((score, maskA if scoreA >= scoreB else maskB))
        else:
            if scoreA >= UNIQUE_SCORE:
                kept.append((scoreA, maskA))
    for j, (scoreB, maskB) in enumerate(candsB):
        if not usedB[j] and scoreB >= UNIQUE_SCORE:
            kept.append((scoreB, maskB))
    return kept


@torch.no_grad()
def predict_ensemble(detA, detB, refiner, device, gray_img, img_t):
    candsA = raw_candidates(detA, device, img_t)
    candsB = raw_candidates(detB, device, img_t)
    merged = dedup(candsA, candsB)

    refined = []
    for score, coarse in merged:
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
        if full_mask.sum() == 0:
            continue
        refined.append((score, full_mask))

    return paint_panoptic(refined, MIN_AREA)


def main():
    device = torch.device("cuda")
    stateA = torch.load("checkpoints/maskrcnn_epoch3.pt", map_location=device)
    detA = build_from_checkpoint(stateA, num_classes=2).to(device)
    detA.load_state_dict(stateA["model"]); detA.roi_heads.nms_thresh = 0.30; detA.eval()

    stateB = torch.load("checkpoints/maskrcnn_tile_epoch7.pt", map_location=device)
    detB = build_from_checkpoint(stateB, num_classes=2).to(device)
    detB.load_state_dict(stateB["model"]); detB.roi_heads.nms_thresh = 0.30; detB.eval()

    rstate = torch.load("checkpoints/refiner_v5_best.pt", map_location=device)
    refiner = RefinerUNet(in_channels=rstate.get("in_channels", 1),
                           out_channels=rstate.get("out_channels", 1)).to(device)
    refiner.load_state_dict(rstate["model"]); refiner.eval()

    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)

    tp = fp_near = fp_spur = fn_near = fn_total = 0
    tp_ious = []
    pq_num = pq_den = 0.0
    for i, e in enumerate(val_entries, 1):
        gray = np.array(Image.open(IMG_DIR / e["file_name"]).convert("L"))
        rgb = np.array(Image.open(IMG_DIR / e["file_name"]).convert("RGB"))
        img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        kept = predict_ensemble(detA, detB, refiner, device, gray, img_t)
        pred = [{"size": [H, W], "counts": to_rle(m).encode()} for m in kept]

        gt = []
        for a in per_image.get(e["id"], []):
            rles = mu.frPyObjects(a["segmentation"], H, W)
            gt.append({"size": [H, W], "counts": to_rle(mu.decode(mu.merge(rles))).encode()})

        if pred and gt:
            iou = mu.iou(pred, gt, [0] * len(gt))
            best_per_gt = iou.max(axis=0); best_per_pred = iou.max(axis=1)
        else:
            best_per_gt = np.zeros(len(gt)); best_per_pred = np.zeros(len(pred))

        m_tp = int((best_per_gt > 0.5).sum())
        tp += m_tp
        tp_ious.extend(best_per_gt[best_per_gt > 0.5].tolist())
        fn_near += int(((best_per_gt > 0.2) & (best_per_gt <= 0.5)).sum())
        fn_total += int((best_per_gt <= 0.2).sum())
        fp_near += int(((best_per_pred > 0.2) & (best_per_pred <= 0.5)).sum())
        fp_spur += int((best_per_pred <= 0.2).sum())

        n_fp = len(pred) - m_tp
        n_fn = len(gt) - m_tp
        pq_num += float(best_per_gt[best_per_gt > 0.5].sum())
        pq_den += m_tp + 0.5 * n_fp + 0.5 * n_fn
        if i % 10 == 0:
            print(f"  {i}/{len(val_entries)}", flush=True)

    n_gt = tp + fn_near + fn_total
    n_pred = tp + fp_near + fp_spur
    print(f"\n=== {len(val_entries)} val entries, {n_gt} GT filaments, {n_pred} predictions ===")
    print(f"TP            : {tp:4d}  ({100*tp/n_gt:.0f}% of GT)   mean IoU of TPs = {np.mean(tp_ious):.3f}")
    print(f"FN near-miss  : {fn_near:4d}  ({100*fn_near/n_gt:.0f}% of GT)")
    print(f"FN total-miss : {fn_total:4d}  ({100*fn_total/n_gt:.0f}% of GT)")
    print(f"FP near-miss  : {fp_near:4d}  ({100*fp_near/n_pred:.0f}% of preds)")
    print(f"FP spurious   : {fp_spur:4d}  ({100*fp_spur/n_pred:.0f}% of preds)")
    print(f"aggregate PQ  : {pq_num/pq_den:.3f}")


if __name__ == "__main__":
    main()
