"""Two-stage inference: YOLO11s-seg detector -> crop-refine U-Net.

A genuinely different detector architecture (anchor-free, different
backbone/FPN) from the Mask R-CNN family used everywhere else this
session, feeding into the SAME proven crop-refine U-Net refiner + TTA.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from ultralytics import YOLO
import pycocotools.mask as mu

from dataset import H, W, IMG_DIR, train_val_split
from train_refiner import RefinerUNet
from crop_dataset import square_bounds, CROP_SIZE
from predict_trained import paint_panoptic, to_rle
from predict_refined import refine_with_tta

D = Path("data/MAGFiLO_1.0_Kaggle_2026")
TEST_DIR = D / "test" / "test_images"

YOLO_CONF = 0.25
MIN_AREA = 20
CHUNK_SIZE = 20
COOLDOWN_SECONDS = 15


def load_models(yolo_ckpt: str, refiner_ckpt: str, device):
    yolo = YOLO(yolo_ckpt)
    rstate = torch.load(refiner_ckpt, map_location=device)
    refiner = RefinerUNet(in_channels=rstate.get("in_channels", 1),
                           out_channels=rstate.get("out_channels", 1)).to(device)
    refiner.load_state_dict(rstate["model"])
    refiner.eval()
    return yolo, refiner


@torch.no_grad()
def predict_one(yolo, refiner, device, gray: np.ndarray, img_path):
    out = yolo.predict(source=str(img_path), imgsz=1280, conf=YOLO_CONF, verbose=False)[0]
    if out.boxes is None or len(out.boxes) == 0:
        return []
    boxes = out.boxes.xyxy.cpu().numpy()
    scores = out.boxes.conf.cpu().numpy()

    candidates = []
    for j in range(len(boxes)):
        x0, y0, x1, y1 = boxes[j]
        x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
        if x1 <= x0 or y1 <= y0:
            continue
        coarse = np.zeros((H, W), dtype=np.uint8)
        coarse[y0:y1, x0:x1] = 1

        cx0, cy0, cx1, cy1 = square_bounds(coarse.astype(bool))
        crop = np.array(Image.fromarray(gray[cy0:cy1, cx0:cx1]).resize(
            (CROP_SIZE, CROP_SIZE), Image.BILINEAR)).astype(np.float32) / 255.0
        prob = refine_with_tta(refiner, device, np.stack([crop]))
        side = cy1 - cy0
        prob_full = np.array(Image.fromarray((prob * 255).astype(np.uint8)).resize(
            (side, side), Image.BILINEAR)).astype(np.float32) / 255.0
        refined_crop = (prob_full > 0.5).astype(np.uint8)
        full_mask = np.zeros((H, W), dtype=np.uint8)
        full_mask[cy0:cy1, cx0:cx1] = refined_crop
        if full_mask.sum() == 0:
            continue
        candidates.append((float(scores[j]), full_mask))

    return paint_panoptic(candidates, MIN_AREA)


def eval_val(yolo_ckpt: str, refiner_ckpt: str, n: int = 116):
    from pq import compute_pq
    device = torch.device("cuda")
    yolo, refiner = load_models(yolo_ckpt, refiner_ckpt, device)
    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)

    tp = fp_near = fp_spur = fn_near = fn_total = 0
    tp_ious = []
    pq_num = pq_den = 0.0
    for i, e in enumerate(val_entries[:n], 1):
        img_path = IMG_DIR / e["file_name"]
        gray = np.array(Image.open(img_path).convert("L"))
        kept = predict_one(yolo, refiner, device, gray, img_path)
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
            print(f"  {i}/{n}", flush=True)

    n_gt = tp + fn_near + fn_total
    n_pred = tp + fp_near + fp_spur
    print(f"\n=== {n} val entries, {n_gt} GT filaments, {n_pred} predictions (YOLO) ===")
    print(f"TP            : {tp:4d}  ({100*tp/n_gt:.0f}% of GT)   mean IoU of TPs = {np.mean(tp_ious):.3f}")
    print(f"FN near-miss  : {fn_near:4d}  ({100*fn_near/n_gt:.0f}% of GT)")
    print(f"FN total-miss : {fn_total:4d}  ({100*fn_total/n_gt:.0f}% of GT)")
    print(f"FP near-miss  : {fp_near:4d}  ({100*fp_near/n_pred:.0f}% of preds)")
    print(f"FP spurious   : {fp_spur:4d}  ({100*fp_spur/n_pred:.0f}% of preds)")
    print(f"aggregate PQ  : {pq_num/pq_den:.3f}")


def predict(yolo_ckpt: str, refiner_ckpt: str, out_csv: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo, refiner = load_models(yolo_ckpt, refiner_ckpt, device)
    print(f"loaded yolo={yolo_ckpt} refiner={refiner_ckpt} on {device}")

    partial_path = Path(out_csv + ".partial")
    files = sorted(TEST_DIR.iterdir())

    done_stems = set()
    rows = []
    if partial_path.exists():
        prev = pd.read_csv(partial_path, dtype=str)
        rows = prev.to_dict("records")
        done_stems = set(prev["filament_id"].str.rsplit("_", n=1).str[0])
        print(f"resuming: {len(done_stems)} images already done")

    remaining = [p for p in files if p.stem not in done_stems]
    for chunk_start in range(0, len(remaining), CHUNK_SIZE):
        chunk = remaining[chunk_start:chunk_start + CHUNK_SIZE]
        for path in chunk:
            gray = np.array(Image.open(path).convert("L"))
            kept = predict_one(yolo, refiner, device, gray, path)
            stem = path.stem
            rows.extend({"filament_id": f"{stem}_{k}", "segmentation_rle": to_rle(m)}
                        for k, m in enumerate(kept, 1))
            print(f"{stem}: {len(kept)} kept", flush=True)

        pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(
            partial_path, index=False)
        done = min(chunk_start + CHUNK_SIZE, len(remaining))
        print(f"--- chunk done: {done}/{len(remaining)} remaining "
              f"({len(files) - len(remaining) + done}/{len(files)} total) ---", flush=True)
        if done < len(remaining):
            torch.cuda.empty_cache() if device.type == "cuda" else None
            time.sleep(COOLDOWN_SECONDS)

    partial_path.replace(out_csv)
    out_df = pd.read_csv(out_csv)
    print(f"\nwrote {out_csv}: {len(out_df)} rows, {len(files)} images, "
          f"avg {len(out_df)/len(files):.1f} filaments/image")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--yolo", default="runs/segment/yolo_runs/filament/weights/best.pt")
    p.add_argument("--refiner", default="checkpoints/refiner_v5_best.pt")
    p.add_argument("--n", type=int, default=116)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.out:
        predict(args.yolo, args.refiner, args.out)
    else:
        eval_val(args.yolo, args.refiner, args.n)
