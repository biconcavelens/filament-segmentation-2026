"""Error decomposition for the CURRENT BEST pipeline (predict_ensemble_dedup.py,
real 0.39) -- answers "are we stuck at the model, or is there genuinely no
room left": total-miss (best IoU<=0.2, a detector recall problem, in
principle fixable) vs near-miss (0.2<IoU<=0.5, boundary/ambiguity, PQ
double-penalizes these) vs spurious FP, against the held-out val split.

If near-miss dominates the remaining gap, that's consistent with running
into genuine annotation ambiguity (this dataset's own docs note 42% of
images have annotators who disagree on which filaments to mark) rather
than a fixable detector weakness -- a real ceiling, not a skill issue.
"""
import numpy as np
import torch
import pycocotools.mask as mu
from PIL import Image
from ultralytics import YOLO

from dataset import train_val_split, IMG_DIR, H, W
from train import build_from_checkpoint
from train_refiner import RefinerUNet
from predict_ensemble_dedup import (
    MASKRCNN_CKPT, YOLO_CKPT, REFINER_CKPT, fit_calibration, predict_one,
)
from predict_trained import to_rle


def main():
    device = torch.device("cuda")
    stateA = torch.load(MASKRCNN_CKPT, map_location=device)
    detA = build_from_checkpoint(stateA, num_classes=2).to(device)
    detA.load_state_dict(stateA["model"])
    detA.roi_heads.nms_thresh = 0.30
    detA.eval()

    yolo = YOLO(YOLO_CKPT)

    rstate = torch.load(REFINER_CKPT, map_location=device)
    refiner = RefinerUNet(in_channels=rstate.get("in_channels", 1),
                           out_channels=rstate.get("out_channels", 1)).to(device)
    refiner.load_state_dict(rstate["model"])
    refiner.eval()

    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)
    print("fitting isotonic calibration on held-out val split...", flush=True)
    calA, calB = fit_calibration(detA, yolo, refiner, device, val_entries, per_image)
    print("calibration fit done.\n", flush=True)

    tp = fp_near = fp_spur = fn_near = fn_total = 0
    tp_ious = []
    pq_num = pq_den = 0.0

    # track which GT filaments are near/total misses for the size/contrast check
    miss_areas = []
    miss_contrasts = []

    with torch.no_grad():
        for i, e in enumerate(val_entries, 1):
            img_path = IMG_DIR / e["file_name"]
            gray = np.array(Image.open(img_path).convert("L"))
            rgb = np.array(Image.open(img_path).convert("RGB"))
            kept = predict_one(detA, yolo, refiner, calA, calB, device, gray, rgb, img_path)
            pred = [{"size": [H, W], "counts": to_rle(m).encode()} for m in kept]

            gt = []
            gt_polys = []
            for a in per_image.get(e["id"], []):
                rles = mu.frPyObjects(a["segmentation"], H, W)
                m = mu.decode(mu.merge(rles))
                gt.append({"size": [H, W], "counts": to_rle(m).encode()})
                gt_polys.append(m)

            if pred and gt:
                iou = mu.iou(pred, gt, [0] * len(gt))
                best_per_gt = iou.max(axis=0)
                best_per_pred = iou.max(axis=1)
            else:
                best_per_gt = np.zeros(len(gt))
                best_per_pred = np.zeros(len(pred))

            m_tp = int((best_per_gt > 0.5).sum())
            tp += m_tp
            tp_ious.extend(best_per_gt[best_per_gt > 0.5].tolist())
            fn_near += int(((best_per_gt > 0.2) & (best_per_gt <= 0.5)).sum())
            fn_total += int((best_per_gt <= 0.2).sum())
            fp_near += int(((best_per_pred > 0.2) & (best_per_pred <= 0.5)).sum())
            fp_spur += int((best_per_pred <= 0.2).sum())

            for j, m in enumerate(gt_polys):
                if best_per_gt[j] <= 0.5:
                    area = int(m.sum())
                    ys, xs = np.where(m)
                    if len(ys) == 0:
                        continue
                    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
                    pad = 5
                    inner = gray[max(0, y0):y1, max(0, x0):x1].astype(np.float32)
                    ring = gray[max(0, y0-pad):y1+pad, max(0, x0-pad):x1+pad].astype(np.float32)
                    contrast = float(ring.mean() - inner.mean()) if inner.size else 0.0
                    miss_areas.append(area)
                    miss_contrasts.append(contrast)

            n_fp = len(pred) - m_tp
            n_fn = len(gt) - m_tp
            pq_num += float(best_per_gt[best_per_gt > 0.5].sum())
            pq_den += m_tp + 0.5 * n_fp + 0.5 * n_fn
            if i % 20 == 0:
                print(f"  {i}/{len(val_entries)}", flush=True)

    n_gt = tp + fn_near + fn_total
    n_pred = tp + fp_near + fp_spur
    print(f"\n=== {len(val_entries)} val images, {n_gt} GT filaments, {n_pred} predictions ===")
    print(f"TP            : {tp:4d}  ({100*tp/n_gt:.0f}% of GT)   mean IoU of TPs = {np.mean(tp_ious):.3f}")
    print(f"FN near-miss  : {fn_near:4d}  ({100*fn_near/n_gt:.0f}% of GT)  -- boundary/ambiguity, PQ-double-penalized")
    print(f"FN total-miss : {fn_total:4d}  ({100*fn_total/n_gt:.0f}% of GT)  -- detector never proposed anything here")
    print(f"FP near-miss  : {fp_near:4d}  ({100*fp_near/n_pred:.0f}% of preds)")
    print(f"FP spurious   : {fp_spur:4d}  ({100*fp_spur/n_pred:.0f}% of preds)")
    print(f"aggregate PQ  : {pq_num/pq_den:.4f}")

    if miss_areas:
        miss_areas = np.array(miss_areas)
        miss_contrasts = np.array(miss_contrasts)
        print(f"\n=== missed filaments (near+total, n={len(miss_areas)}) ===")
        print(f"area:     median={np.median(miss_areas):.0f}px  mean={miss_areas.mean():.0f}px")
        print(f"contrast: median={np.median(miss_contrasts):.1f}  mean={miss_contrasts.mean():.1f} gray levels")
        print("(compare to caught filaments' typical area ~1400px, contrast ~16.5 gray levels")
        print(" from earlier diag_missed_filaments.py -- if misses skew much smaller/fainter,")
        print(" that's consistent with hitting genuine annotation-ambiguity limits)")


if __name__ == "__main__":
    main()
