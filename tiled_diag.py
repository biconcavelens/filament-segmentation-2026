"""Does the detector miss filaments because they're sub-pixel after the
2048->~800px training-time resize? Test tiled inference with the EXISTING
trained detector (no retraining): split each image into overlapping tiles,
run detection per-tile (each tile's own transform.min_size/max_size resize
is a gentler downscale than the whole-image resize), map boxes/masks back to
full-image coordinates, dedup overlap-zone duplicates, then run the same
crop-refine + TTA pipeline as the single-pass baseline. Compares recall
(TP, FN-total-miss) directly against the un-tiled baseline on full val.
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

TILE = 1152
OVERLAP = 256
STEP = TILE - OVERLAP
DEDUP_IOU = 0.5


def tile_origins(size, tile, step):
    origins = list(range(0, size - tile + 1, step))
    if origins[-1] != size - tile:
        origins.append(size - tile)
    return origins


@torch.no_grad()
def tiled_candidates(detector, device, rgb):
    cands = []
    for y0 in tile_origins(H, TILE, STEP):
        for x0 in tile_origins(W, TILE, STEP):
            tile = rgb[y0:y0 + TILE, x0:x0 + TILE]
            img_t = torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0
            out = detector([img_t.to(device)])[0]
            scores = out["scores"].cpu().numpy()
            masks = out["masks"].cpu().numpy()
            for j in range(len(scores)):
                if scores[j] < DETECTOR_SCORE_THRESHOLD:
                    continue
                coarse_tile = (masks[j, 0] > DETECTOR_MASK_THRESHOLD).astype(np.uint8)
                if coarse_tile.sum() == 0:
                    continue
                full = np.zeros((H, W), dtype=np.uint8)
                full[y0:y0 + TILE, x0:x0 + TILE] = coarse_tile
                cands.append((float(scores[j]), full))
    return cands


def dedup(cands):
    cands = sorted(cands, key=lambda x: -x[0])
    kept = []
    for score, mask in cands:
        dup = False
        for ks, km in kept:
            inter = np.logical_and(mask, km).sum()
            if inter == 0:
                continue
            union = np.logical_or(mask, km).sum()
            if inter / union > DEDUP_IOU:
                dup = True
                break
        if not dup:
            kept.append((score, mask))
    return kept


@torch.no_grad()
def predict_tiled(detector, refiner, device, gray, rgb):
    cands = dedup(tiled_candidates(detector, device, rgb))
    refined = []
    for score, coarse in cands:
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
        if full_mask.sum() == 0:
            continue
        refined.append((score, full_mask))
    return paint_panoptic(refined, MIN_AREA)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--detector", default="checkpoints/maskrcnn_epoch3.pt")
    p.add_argument("--n", type=int, default=116)
    args = p.parse_args()

    device = torch.device("cuda")
    state = torch.load(args.detector, map_location=device)
    detector = build_from_checkpoint(state, num_classes=2).to(device)
    detector.load_state_dict(state["model"]); detector.roi_heads.nms_thresh = 0.30; detector.eval()

    rstate = torch.load("checkpoints/refiner_best.pt", map_location=device)
    refiner = RefinerUNet(in_channels=rstate.get("in_channels", 1)).to(device)
    refiner.load_state_dict(rstate["model"]); refiner.eval()

    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)
    val_entries = val_entries[:args.n]

    tp = fp_near = fp_spur = fn_near = fn_total = 0
    tp_ious = []
    pq_num = pq_den = 0.0
    for i, e in enumerate(val_entries, 1):
        gray = np.array(Image.open(IMG_DIR / e["file_name"]).convert("L"))
        rgb = np.array(Image.open(IMG_DIR / e["file_name"]).convert("RGB"))
        kept = predict_tiled(detector, refiner, device, gray, rgb)
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
    print(f"\n=== {len(val_entries)} val entries, {n_gt} GT filaments, {n_pred} predictions (TILED) ===")
    print(f"TP            : {tp:4d}  ({100*tp/n_gt:.0f}% of GT)   mean IoU of TPs = {np.mean(tp_ious):.3f}")
    print(f"FN near-miss  : {fn_near:4d}  ({100*fn_near/n_gt:.0f}% of GT)")
    print(f"FN total-miss : {fn_total:4d}  ({100*fn_total/n_gt:.0f}% of GT)")
    print(f"FP near-miss  : {fp_near:4d}  ({100*fp_near/n_pred:.0f}% of preds)")
    print(f"FP spurious   : {fp_spur:4d}  ({100*fp_spur/n_pred:.0f}% of preds)")
    print(f"aggregate PQ  : {pq_num/pq_den:.3f}")


if __name__ == "__main__":
    main()
