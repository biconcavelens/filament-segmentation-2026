"""Cache raw watershed model outputs (semantic+distance maps) once, then
sweep SEMANTIC_THRESHOLD / SPLIT_AREA_THRESHOLD / CORE_FRACTION cheaply
against the cache -- FP-spurious was 54% of all predictions at the default
0.5 threshold, so the semantic head is likely just too permissive.
"""
import numpy as np
import torch
import pycocotools.mask as mu
from PIL import Image

from dataset import H, W, IMG_DIR, train_val_split
from train_watershed import WatershedUNet
from predict_watershed import predict_maps, instances_from_maps, to_rle
import predict_watershed as pw

device = torch.device("cuda")
model = WatershedUNet().to(device)
state = torch.load("checkpoints/watershed_best.pt", map_location=device)
model.load_state_dict(state["model"]); model.eval()

_, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)

cache = []
for i, e in enumerate(val_entries, 1):
    gray = np.array(Image.open(IMG_DIR / e["file_name"]).convert("L"))
    sem_full, dist_full = predict_maps(model, device, gray)
    gt = []
    for a in per_image.get(e["id"], []):
        rles = mu.frPyObjects(a["segmentation"], H, W)
        gt.append(to_rle(mu.decode(mu.merge(rles))))
    cache.append((sem_full, dist_full, gt))
    if i % 20 == 0:
        print(f"  cached {i}/{len(val_entries)}", flush=True)

print("cache built, sweeping thresholds...")


def pq_for(sem_thresh, split_area, core_frac):
    pw.SEMANTIC_THRESHOLD = sem_thresh
    pw.SPLIT_AREA_THRESHOLD = split_area
    pw.CORE_FRACTION = core_frac
    pq_num = pq_den = 0.0
    tp = fp_spur = 0
    n_pred_total = 0
    for sem_full, dist_full, gt in cache:
        kept = instances_from_maps(sem_full, dist_full)
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
for sem_thresh in [0.5, 0.7, 0.85, 0.93, 0.97]:
    pq, tp, fp_spur, n_pred = pq_for(sem_thresh, 4500, 0.55)
    print(f"sem_thresh={sem_thresh}: PQ={pq:.4f} TP={tp} FP_spur={fp_spur} n_pred={n_pred}", flush=True)
    results.append((pq, sem_thresh))

results.sort(reverse=True)
best_thresh = results[0][1]
print(f"\nbest sem_thresh so far: {best_thresh} (PQ={results[0][0]:.4f})")

print("\nsweeping split_area at best sem_thresh...")
for split_area in [2500, 4500, 7000, 100000]:
    pq, tp, fp_spur, n_pred = pq_for(best_thresh, split_area, 0.55)
    print(f"split_area={split_area}: PQ={pq:.4f} TP={tp} FP_spur={fp_spur} n_pred={n_pred}", flush=True)
