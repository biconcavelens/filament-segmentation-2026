"""Ensemble YOLO11s-seg with the existing Mask R-CNN detector, using the
same agreement-weighted merge as ensemble_diag.py. These two are
architecturally about as different as two object detectors get (anchor-free
vs. anchor-based, different backbone/FPN, different loss/NMS), so their
errors are more likely to be genuinely complementary than two Mask R-CNN
variants trained on overlapping data.
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
from predict_trained import paint_panoptic, to_rle
from predict_refined import refine_with_tta, DETECTOR_MASK_THRESHOLD
from ensemble_diag import raw_candidates as maskrcnn_raw_candidates

YOLO_FLOOR = 0.15  # loose floor for cross-detector matching visibility, matches the Mask R-CNN FLOOR_SCORE spirit
CROSS_DEDUP_IOU = 0.5
# Each detector's confidence score is on its OWN scale (different loss/
# calibration) -- a "unique" (only one detector saw it) candidate should be
# judged against THAT detector's own already-validated solo operating
# point, not a threshold borrowed from a different architecture:
MASKRCNN_UNIQUE = 0.92   # stricter than the solo bar: in a union, each detector's OWN noise no
YOLO_UNIQUE = 0.55       # longer competes against the other's, so unconfirmed candidates need to
                         # individually clear a much higher bar than when deployed alone
AGREE_SCORE = 0.5        # two independently-calibrated architectures corroborating is strong evidence either way


def dedup(candsA, candsB):
    """candsA = Mask R-CNN candidates, candsB = YOLO candidates."""
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
            if iou > CROSS_DEDUP_IOU and iou > best_iou:
                best_j, best_iou = j, iou
        if best_j >= 0:
            usedB[best_j] = True
            scoreB, maskB = candsB[best_j]
            if max(scoreA, scoreB) >= AGREE_SCORE:
                kept.append((max(scoreA, scoreB), maskA if scoreA >= scoreB else maskB))
        else:
            if scoreA >= MASKRCNN_UNIQUE:
                kept.append((scoreA, maskA))
    for j, (scoreB, maskB) in enumerate(candsB):
        if not usedB[j] and scoreB >= YOLO_UNIQUE:
            kept.append((scoreB, maskB))
    return kept


@torch.no_grad()
def yolo_raw_candidates(yolo, img_path, floor=YOLO_FLOOR):
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
def predict_ensemble(detA, yolo, refiner, device, gray_img, img_t, img_path):
    candsA = maskrcnn_raw_candidates(detA, device, img_t)
    candsB = yolo_raw_candidates(yolo, img_path)
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

    return paint_panoptic(refined, 20)


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

    tp = fp_near = fp_spur = fn_near = fn_total = 0
    tp_ious = []
    pq_num = pq_den = 0.0
    for i, e in enumerate(val_entries, 1):
        img_path = IMG_DIR / e["file_name"]
        gray = np.array(Image.open(img_path).convert("L"))
        rgb = np.array(Image.open(img_path).convert("RGB"))
        img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        kept = predict_ensemble(detA, yolo, refiner, device, gray, img_t, img_path)
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
    print(f"\n=== {len(val_entries)} val entries, {n_gt} GT filaments, {n_pred} predictions (YOLO+MaskRCNN) ===")
    print(f"TP            : {tp:4d}  ({100*tp/n_gt:.0f}% of GT)   mean IoU of TPs = {np.mean(tp_ious):.3f}")
    print(f"FN near-miss  : {fn_near:4d}  ({100*fn_near/n_gt:.0f}% of GT)")
    print(f"FN total-miss : {fn_total:4d}  ({100*fn_total/n_gt:.0f}% of GT)")
    print(f"FP near-miss  : {fp_near:4d}  ({100*fp_near/n_pred:.0f}% of preds)")
    print(f"FP spurious   : {fp_spur:4d}  ({100*fp_spur/n_pred:.0f}% of preds)")
    print(f"aggregate PQ  : {pq_num/pq_den:.3f}")


if __name__ == "__main__":
    main()
