"""Three-detector candidate-union ensemble: fold a third independently
trained checkpoint (the copy-paste-augmented one, already on disk, no new
training needed) into the same agreement-weighted merge that took the
two-detector ensemble from 0.390->0.404 local PQ. Applied pairwise:
merge(A,B) first, then merge(that, C) with the same agree/unique thresholds.
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
from predict_refined import refine_with_tta, MIN_AREA
from ensemble_diag import raw_candidates, dedup


@torch.no_grad()
def predict_ensemble3(detA, detB, detC, refiner, device, gray_img, img_t):
    candsA = raw_candidates(detA, device, img_t)
    candsB = raw_candidates(detB, device, img_t)
    candsC = raw_candidates(detC, device, img_t)
    merged_ab = dedup(candsA, candsB)
    merged = dedup(merged_ab, candsC)

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


def load_det(ckpt, device):
    state = torch.load(ckpt, map_location=device)
    det = build_from_checkpoint(state, num_classes=2).to(device)
    det.load_state_dict(state["model"]); det.roi_heads.nms_thresh = 0.30; det.eval()
    return det


def main():
    device = torch.device("cuda")
    detA = load_det("checkpoints/maskrcnn_epoch3.pt", device)
    detB = load_det("checkpoints/maskrcnn_tile_epoch7.pt", device)
    detC = load_det("checkpoints/maskrcnn_cp_epoch7.pt", device)

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
        kept = predict_ensemble3(detA, detB, detC, refiner, device, gray, img_t)
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
    print(f"\n=== {len(val_entries)} val entries, {n_gt} GT filaments, {n_pred} predictions (3-way) ===")
    print(f"TP            : {tp:4d}  ({100*tp/n_gt:.0f}% of GT)   mean IoU of TPs = {np.mean(tp_ious):.3f}")
    print(f"FN near-miss  : {fn_near:4d}  ({100*fn_near/n_gt:.0f}% of GT)")
    print(f"FN total-miss : {fn_total:4d}  ({100*fn_total/n_gt:.0f}% of GT)")
    print(f"FP near-miss  : {fp_near:4d}  ({100*fp_near/n_pred:.0f}% of preds)")
    print(f"FP spurious   : {fp_spur:4d}  ({100*fp_spur/n_pred:.0f}% of preds)")
    print(f"aggregate PQ  : {pq_num/pq_den:.3f}")


if __name__ == "__main__":
    main()
