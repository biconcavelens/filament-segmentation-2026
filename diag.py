"""Error decomposition for the current best pipeline against local val.

PQ double-penalises a near miss (IoU in (0.2, 0.5]): it counts as an FP *and*
an FN with zero credit. So the two error classes need different fixes:
  total miss  (best IoU <= 0.2)     -> detector recall / anchors / resolution
  near miss   (0.2 < best IoU <= 0.5) -> refiner boundary quality
This script reports which one dominates so we stop guessing.

    DET_NMS=0.7 python diag.py      # optional NMS override on the detector
"""
import argparse
import os
import time

import numpy as np
import torch
import pycocotools.mask as mu
from PIL import Image

from dataset import train_val_split, IMG_DIR, H, W
from predict_refined import load_models, predict_one, to_rle

N = 30


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--detector", default="checkpoints/maskrcnn_epoch3.pt")
    p.add_argument("--refiner", default="checkpoints/refiner_best.pt")
    p.add_argument("--n", type=int, default=N)
    args = p.parse_args()

    device = torch.device("cuda")
    detector, refiner = load_models(args.detector, args.refiner, device)
    nms = os.environ.get("DET_NMS")
    if nms:
        detector.roi_heads.nms_thresh = float(nms)
        print(f"detector NMS thresh overridden to {nms}")

    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)

    tp = fp_near = fp_spur = fn_near = fn_total = 0
    tp_ious = []
    pq_num = pq_den = 0.0
    for i, e in enumerate(val_entries[:args.n], 1):
        gray = np.array(Image.open(IMG_DIR / e["file_name"]).convert("L"))
        rgb = np.array(Image.open(IMG_DIR / e["file_name"]).convert("RGB"))
        img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        kept = predict_one(detector, refiner, device, gray, img_t)
        pred = [{"size": [H, W], "counts": to_rle(m).encode()} for m in kept]

        gt = []
        for a in per_image.get(e["id"], []):
            rles = mu.frPyObjects(a["segmentation"], H, W)
            gt.append({"size": [H, W], "counts": to_rle(mu.decode(mu.merge(rles))).encode()})

        if pred and gt:
            iou = mu.iou(pred, gt, [0] * len(gt))  # [n_pred, n_gt]
            best_per_gt = iou.max(axis=0)
            best_per_pred = iou.max(axis=1)
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
            print(f"  {i}/{args.n}", flush=True)

    n_gt = tp + fn_near + fn_total
    n_pred = tp + fp_near + fp_spur
    print(f"\n=== {args.n} val entries, {n_gt} GT filaments, {n_pred} predictions ===")
    print(f"TP            : {tp:4d}  ({100*tp/n_gt:.0f}% of GT)   mean IoU of TPs = {np.mean(tp_ious):.3f}")
    print(f"FN near-miss  : {fn_near:4d}  ({100*fn_near/n_gt:.0f}% of GT)   detected but IoU in (0.2,0.5]")
    print(f"FN total-miss : {fn_total:4d}  ({100*fn_total/n_gt:.0f}% of GT)   never found (IoU<=0.2)")
    print(f"FP near-miss  : {fp_near:4d}  ({100*fp_near/n_pred:.0f}% of preds) loose boundary on a real filament")
    print(f"FP spurious   : {fp_spur:4d}  ({100*fp_spur/n_pred:.0f}% of preds) not a filament (IoU<=0.2)")
    print(f"aggregate PQ  : {pq_num/pq_den:.3f}")


if __name__ == "__main__":
    main()
