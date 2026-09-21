"""Isolated refiner comparison: same YOLO11 detector candidates (cached
once), refined by v5 (spine-only, deployed) vs v6 (spine+clDice, new) --
tests whether clDice moves PQ without re-running detector inference,
mirroring the original spine-weight sweep's experimental design (vary
only the refiner, hold the detector fixed).
"""
import numpy as np
import torch
import pycocotools.mask as mu
from PIL import Image
from ultralytics import YOLO

from dataset import train_val_split, IMG_DIR, H, W
from train_refiner import RefinerUNet
from crop_dataset import square_bounds, CROP_SIZE
from predict_trained import to_rle
from predict_refined import refine_with_tta

YOLO_CKPT = "kaggle_dataset_upload/yolo11m_cls_best.pt"
REFINER_V5 = "checkpoints/refiner_v5_best.pt"
REFINER_V6 = "checkpoints/refiner_v6_best.pt"
YOLO_CONF = 0.33  # validated solo YOLO11 operating point
MIN_AREA = 20


@torch.no_grad()
def refine_candidate(refiner, device, gray, coarse):
    x0, y0, x1, y1 = square_bounds(coarse.astype(bool))
    crop = np.array(Image.fromarray(gray[y0:y1, x0:x1]).resize(
        (CROP_SIZE, CROP_SIZE), Image.BILINEAR)).astype(np.float32) / 255.0
    prob = refine_with_tta(refiner, device, np.stack([crop]))
    side = y1 - y0
    prob_full = np.array(Image.fromarray((prob * 255).astype(np.uint8)).resize(
        (side, side), Image.BILINEAR)).astype(np.float32) / 255.0
    refined_crop = (prob_full > 0.5).astype(np.uint8)
    full_mask = np.zeros((H, W), dtype=np.uint8)
    full_mask[y0:y1, x0:x1] = refined_crop
    return full_mask


def paint_panoptic(candidates, min_area=MIN_AREA):
    candidates = sorted(candidates, key=lambda x: -x[0])
    claimed = np.zeros((H, W), dtype=np.uint8)
    kept = []
    for score, binary in candidates:
        remaining = binary & (1 - claimed)
        if int(remaining.sum()) < min_area:
            continue
        claimed |= remaining
        kept.append(remaining)
    return kept


def pq_against_gt(kept, gt_dicts):
    pred = [{"size": [H, W], "counts": to_rle(m).encode()} for m in kept]
    if pred and gt_dicts:
        iou = mu.iou(pred, gt_dicts, [0] * len(gt_dicts))
        best_per_gt = iou.max(axis=0)
    else:
        best_per_gt = np.zeros(len(gt_dicts))
    m_tp = int((best_per_gt > 0.5).sum())
    n_fp = len(pred) - m_tp
    n_fn = len(gt_dicts) - m_tp
    num = float(best_per_gt[best_per_gt > 0.5].sum())
    den = m_tp + 0.5 * n_fp + 0.5 * n_fn
    return num, den, m_tp


def load_refiner(ckpt, device):
    rstate = torch.load(ckpt, map_location=device)
    refiner = RefinerUNet(in_channels=rstate.get("in_channels", 1),
                           out_channels=rstate.get("out_channels", 1)).to(device)
    refiner.load_state_dict(rstate["model"])
    refiner.eval()
    return refiner


def main():
    device = torch.device("cuda")
    yolo = YOLO(YOLO_CKPT)
    refiner_v5 = load_refiner(REFINER_V5, device)
    refiner_v6 = load_refiner(REFINER_V6, device)

    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)

    pq_num_v5 = pq_den_v5 = 0.0
    pq_num_v6 = pq_den_v6 = 0.0
    tp_v5 = tp_v6 = 0

    with torch.no_grad():
        for i, e in enumerate(val_entries, 1):
            img_path = IMG_DIR / e["file_name"]
            gray = np.array(Image.open(img_path).convert("L"))

            out = yolo.predict(source=str(img_path), imgsz=1280, conf=YOLO_CONF, verbose=False)[0]
            coarse_cands = []
            if out.boxes is not None and len(out.boxes) > 0:
                boxes = out.boxes.xyxy.cpu().numpy()
                scores = out.boxes.conf.cpu().numpy()
                for j in range(len(boxes)):
                    x0, y0, x1, y1 = boxes[j]
                    x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
                    if x1 <= x0 or y1 <= y0:
                        continue
                    coarse = np.zeros((H, W), dtype=np.uint8)
                    coarse[y0:y1, x0:x1] = 1
                    coarse_cands.append((float(scores[j]), coarse))

            gt = []
            for a in per_image.get(e["id"], []):
                rles = mu.frPyObjects(a["segmentation"], H, W)
                gt.append({"size": [H, W], "counts": to_rle(mu.decode(mu.merge(rles))).encode()})

            candsV5 = [(s, refine_candidate(refiner_v5, device, gray, c)) for s, c in coarse_cands]
            candsV5 = [(s, m) for s, m in candsV5 if m.sum() > 0]
            keptV5 = paint_panoptic(candsV5)
            num, den, m_tp = pq_against_gt(keptV5, gt)
            pq_num_v5 += num; pq_den_v5 += den; tp_v5 += m_tp

            candsV6 = [(s, refine_candidate(refiner_v6, device, gray, c)) for s, c in coarse_cands]
            candsV6 = [(s, m) for s, m in candsV6 if m.sum() > 0]
            keptV6 = paint_panoptic(candsV6)
            num, den, m_tp = pq_against_gt(keptV6, gt)
            pq_num_v6 += num; pq_den_v6 += den; tp_v6 += m_tp

            if i % 20 == 0:
                print(f"  {i}/{len(val_entries)}", flush=True)

    pq_v5 = pq_num_v5 / pq_den_v5 if pq_den_v5 else 0.0
    pq_v6 = pq_num_v6 / pq_den_v6 if pq_den_v6 else 0.0
    print(f"\nrefiner v5 (spine only, deployed): PQ={pq_v5:.4f} TP={tp_v5}")
    print(f"refiner v6 (spine + clDice, new):  PQ={pq_v6:.4f} TP={tp_v6}")
    print(f"delta: {pq_v6 - pq_v5:+.4f}")


if __name__ == "__main__":
    main()
