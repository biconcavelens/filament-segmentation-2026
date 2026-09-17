"""Cheap grid sweep over detector score threshold + NMS thresh, no retraining.
Reuses one loaded model pair across the whole grid instead of reloading per combo.
"""
import numpy as np
import torch
import pycocotools.mask as mu
from PIL import Image

import predict_refined as pr
from dataset import train_val_split, IMG_DIR, H, W

N = 60  # subset of the 116 val images, for speed; winner gets confirmed on full val


def pq_for(detector, refiner, device, val_entries, per_image):
    pq_num = pq_den = 0.0
    for e in val_entries:
        gray = np.array(Image.open(IMG_DIR / e["file_name"]).convert("L"))
        rgb = np.array(Image.open(IMG_DIR / e["file_name"]).convert("RGB"))
        img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        kept = pr.predict_one(detector, refiner, device, gray, img_t)
        pred = [{"size": [H, W], "counts": pr.to_rle(m).encode()} for m in kept]

        gt = []
        for a in per_image.get(e["id"], []):
            rles = mu.frPyObjects(a["segmentation"], H, W)
            gt.append({"size": [H, W], "counts": pr.to_rle(mu.decode(mu.merge(rles))).encode()})

        if pred and gt:
            iou = mu.iou(pred, gt, [0] * len(gt))
            best_per_gt = iou.max(axis=0)
        else:
            best_per_gt = np.zeros(len(gt))

        m_tp = int((best_per_gt > 0.5).sum())
        n_fp = len(pred) - m_tp
        n_fn = len(gt) - m_tp
        pq_num += float(best_per_gt[best_per_gt > 0.5].sum())
        pq_den += m_tp + 0.5 * n_fp + 0.5 * n_fn
    return pq_num / pq_den if pq_den else 0.0


def main():
    device = torch.device("cuda")
    detector, refiner = pr.load_models("checkpoints/maskrcnn_epoch3.pt", "checkpoints/refiner_best.pt", device)
    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)
    subset = val_entries[:N]

    results = []
    for nms in [0.7]:
        detector.roi_heads.nms_thresh = nms
        for thresh in [0.5, 0.65, 0.80, 0.90]:
            pr.DETECTOR_SCORE_THRESHOLD = thresh
            pq = pq_for(detector, refiner, device, subset, per_image)
            print(f"nms={nms} thresh={thresh}: PQ={pq:.4f}", flush=True)
            results.append((pq, nms, thresh))

    results.sort(reverse=True)
    print("\n=== top 5 ===")
    for pq, nms, thresh in results[:5]:
        print(f"PQ={pq:.4f}  nms={nms} thresh={thresh}")


if __name__ == "__main__":
    main()
