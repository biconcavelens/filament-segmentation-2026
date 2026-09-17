"""Test a limb-exclusion post-filter: MAGFiLO's paper states annotators were
told to ignore filaments near the limb (~70deg from central meridian), and
this is empirically an EXACT rule in our training data -- zero GT filament
centroids exist beyond r/R=0.8515 of the disk center (checked over all 8199
training filaments). Disk center/radius are also fixed constants across
every image (1024,1024 / r=1020), confirmed on a 15-image sample. So any
prediction centroid beyond that radius is guaranteed to never match a GT
instance -- dropping it costs zero TPs and can only reduce FP-spurious.
"""
import numpy as np
import torch
import pycocotools.mask as mu
from PIL import Image

from dataset import train_val_split, IMG_DIR, H, W
from train import build_from_checkpoint
from train_refiner import RefinerUNet
from predict_trained import to_rle
from ensemble_diag import predict_ensemble

DISK_CENTER = (1024.0, 1024.0)
DISK_RADIUS = 1020.0
LIMB_CUTOFF = 0.87  # small margin over the observed 0.8515 max


def filter_limb(masks, cutoff=LIMB_CUTOFF):
    kept = []
    for m in masks:
        ys, xs = np.where(m)
        if len(xs) == 0:
            continue
        cx, cy = xs.mean(), ys.mean()
        r = np.hypot(cx - DISK_CENTER[0], cy - DISK_CENTER[1]) / DISK_RADIUS
        if r <= cutoff:
            kept.append(m)
    return kept


def main():
    device = torch.device("cuda")
    stateA = torch.load("checkpoints/maskrcnn_epoch3.pt", map_location=device)
    detA = build_from_checkpoint(stateA, num_classes=2).to(device)
    detA.load_state_dict(stateA["model"]); detA.roi_heads.nms_thresh = 0.30; detA.eval()

    stateB = torch.load("checkpoints/maskrcnn_tile_epoch7.pt", map_location=device)
    detB = build_from_checkpoint(stateB, num_classes=2).to(device)
    detB.load_state_dict(stateB["model"]); detB.roi_heads.nms_thresh = 0.30; detB.eval()

    rstate = torch.load("checkpoints/refiner_best.pt", map_location=device)
    refiner = RefinerUNet(in_channels=rstate.get("in_channels", 1)).to(device)
    refiner.load_state_dict(rstate["model"]); refiner.eval()

    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)

    # run the (expensive) ensemble pipeline ONCE per image, cache raw masks +
    # GT, then score with/without the filter from the same cached predictions
    cache = []
    for i, e in enumerate(val_entries, 1):
        gray = np.array(Image.open(IMG_DIR / e["file_name"]).convert("L"))
        rgb = np.array(Image.open(IMG_DIR / e["file_name"]).convert("RGB"))
        img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        kept = predict_ensemble(detA, detB, refiner, device, gray, img_t)
        gt = []
        for a in per_image.get(e["id"], []):
            rles = mu.frPyObjects(a["segmentation"], H, W)
            gt.append(to_rle(mu.decode(mu.merge(rles))))
        cache.append((kept, gt))
        if i % 20 == 0:
            print(f"  {i}/{len(val_entries)}", flush=True)

    def evaluate(apply_filter):
        tp = fp_near = fp_spur = fn_near = fn_total = 0
        tp_ious = []
        pq_num = pq_den = 0.0
        n_dropped = 0
        for kept, gt_rles in cache:
            masks = kept
            if apply_filter:
                before = len(masks)
                masks = filter_limb(masks)
                n_dropped += before - len(masks)
            pred = [{"size": [H, W], "counts": to_rle(m).encode()} for m in masks]
            gt = [{"size": [H, W], "counts": r.encode()} for r in gt_rles]

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

        n_gt = tp + fn_near + fn_total
        n_pred = tp + fp_near + fp_spur
        label = "WITH limb filter" if apply_filter else "WITHOUT limb filter"
        print(f"\n=== {label}: {n_gt} GT, {n_pred} preds, dropped={n_dropped} ===")
        print(f"TP={tp} FN_near={fn_near} FN_total={fn_total} FP_near={fp_near} FP_spur={fp_spur}")
        print(f"aggregate PQ  : {pq_num/pq_den:.4f}")
        return tp

    tp_before = evaluate(apply_filter=False)
    tp_after = evaluate(apply_filter=True)
    assert tp_after == tp_before, "limb filter must not remove any true positive!"
    print("\nOK: limb filter removed zero true positives, as predicted.")


if __name__ == "__main__":
    main()
