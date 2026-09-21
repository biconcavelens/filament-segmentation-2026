"""Full-test-set prediction: YOLO11(cls=1.5) + RT-DETR(cls=2.0), WITHOUT
Mask R-CNN -- dropping the weakest solo detector (0.4152) in favor of
keeping only the two strongest, structurally-different detectors (CNN
anchor-based vs transformer/DETR-style). Local val PQ 0.4407 (Mask R-CNN+
YOLO11 2-way) -> 0.4440, beating every combination tried this session
except the 4-way (which regressed on the real leaderboard from
overfitting -- this pairing has fewer free parameters/calibrators, so is
a safer bet to transfer). See sweep_ensemble_yolo_rtdetr_2way.py.
"""
import numpy as np
import pandas as pd
import torch
import pycocotools.mask as mu
from PIL import Image
from ultralytics import YOLO, RTDETR
from sklearn.isotonic import IsotonicRegression

from dataset import train_val_split, IMG_DIR, H, W
from train_refiner import RefinerUNet
from crop_dataset import square_bounds, CROP_SIZE
from predict_trained import to_rle
from predict_refined import refine_with_tta

YOLO_CKPT = "kaggle_dataset_upload/yolo11m_cls_best.pt"
RTDETR_CKPT = "kaggle_kernel_rtdetr_cls/output/rtdetr_cls_best.pt"
REFINER_CKPT = "checkpoints/refiner_v5_best.pt"
FLOOR_B, FLOOR_C = 0.05, 0.05
MIN_AREA = 20
DEDUP_IOU = 0.05  # flat plateau 0.03-0.1
ACCEPT_THRESH = 0.45  # genuine interior peak

D = __import__("pathlib").Path("data/MAGFiLO_1.0_Kaggle_2026")
TEST_DIR = D / "test" / "test_images"


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


def best_iou_against_gt(mask, gt_dicts):
    if not gt_dicts:
        return 0.0
    pred_rle = {"size": [H, W], "counts": to_rle(mask).encode()}
    iou = mu.iou([pred_rle], gt_dicts, [0] * len(gt_dicts))
    return float(iou.max())


def dedup_nms(candidates, iou_thresh):
    candidates = sorted(candidates, key=lambda x: -x[0])
    accepted = []
    for score, mask in candidates:
        is_dup = False
        for _, amask in accepted:
            inter = np.logical_and(mask, amask).sum()
            if inter == 0:
                continue
            union = np.logical_or(mask, amask).sum()
            if inter / union > iou_thresh:
                is_dup = True
                break
        if not is_dup:
            accepted.append((score, mask))
    return accepted


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


def _yolo_style_candidates(model, img_path, floor):
    out = model.predict(source=str(img_path), imgsz=1280, conf=floor, verbose=False)[0]
    cands = []
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
            cands.append((float(scores[j]), coarse))
    return cands


def fit_calibration(yolo, rtdetr, refiner, device, val_entries, per_image):
    scoresB_all, labelsB_all = [], []
    scoresC_all, labelsC_all = [], []
    with torch.no_grad():
        for i, e in enumerate(val_entries, 1):
            img_path = IMG_DIR / e["file_name"]
            gray = np.array(Image.open(img_path).convert("L"))

            candsB = _yolo_style_candidates(yolo, img_path, FLOOR_B)
            candsC = _yolo_style_candidates(rtdetr, img_path, FLOOR_C)

            gt = []
            for a in per_image.get(e["id"], []):
                rles = mu.frPyObjects(a["segmentation"], H, W)
                gt.append({"size": [H, W], "counts": to_rle(mu.decode(mu.merge(rles))).encode()})

            for score, coarse in candsB:
                ref = refine_candidate(refiner, device, gray, coarse)
                label = 1 if best_iou_against_gt(ref, gt) > 0.5 else 0
                scoresB_all.append(score)
                labelsB_all.append(label)
            for score, coarse in candsC:
                ref = refine_candidate(refiner, device, gray, coarse)
                label = 1 if best_iou_against_gt(ref, gt) > 0.5 else 0
                scoresC_all.append(score)
                labelsC_all.append(label)
            if i % 20 == 0:
                print(f"  calibration fit: {i}/{len(val_entries)}", flush=True)

    calB = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calB.fit(scoresB_all, labelsB_all)
    calC = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calC.fit(scoresC_all, labelsC_all)
    return calB, calC


@torch.no_grad()
def predict_one(yolo, rtdetr, refiner, calB, calC, device, gray, img_path):
    candsB = _yolo_style_candidates(yolo, img_path, FLOOR_B)
    candsC = _yolo_style_candidates(rtdetr, img_path, FLOOR_C)

    pooled = []
    for score, coarse in candsB:
        ref = refine_candidate(refiner, device, gray, coarse)
        if ref.sum() == 0:
            continue
        cal_score = float(calB.predict([score])[0])
        pooled.append((cal_score, ref))
    for score, coarse in candsC:
        ref = refine_candidate(refiner, device, gray, coarse)
        if ref.sum() == 0:
            continue
        cal_score = float(calC.predict([score])[0])
        pooled.append((cal_score, ref))

    filtered = [(s, m) for s, m in pooled if s >= ACCEPT_THRESH]
    deduped = dedup_nms(filtered, DEDUP_IOU)
    return paint_panoptic(deduped)


def main():
    device = torch.device("cuda")
    yolo = YOLO(YOLO_CKPT)
    rtdetr = RTDETR(RTDETR_CKPT)

    rstate = torch.load(REFINER_CKPT, map_location=device)
    refiner = RefinerUNet(in_channels=rstate.get("in_channels", 1),
                           out_channels=rstate.get("out_channels", 1)).to(device)
    refiner.load_state_dict(rstate["model"])
    refiner.eval()

    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)
    print("fitting isotonic calibration on held-out val split...", flush=True)
    calB, calC = fit_calibration(yolo, rtdetr, refiner, device, val_entries, per_image)
    print("calibration fit done.\n", flush=True)

    files = sorted(TEST_DIR.iterdir())
    rows = []
    for i, path in enumerate(files, 1):
        gray = np.array(Image.open(path).convert("L"))
        kept = predict_one(yolo, rtdetr, refiner, calB, calC, device, gray, path)
        stem = path.stem
        rows.extend({"filament_id": f"{stem}_{k}", "segmentation_rle": to_rle(m)}
                    for k, m in enumerate(kept, 1))
        if i % 20 == 0:
            print(f"{i}/{len(files)} test images done", flush=True)

    out_df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    out_df.to_csv("submission_yolo_rtdetr.csv", index=False)
    print(f"\nwrote submission_yolo_rtdetr.csv: {len(out_df)} rows, {len(files)} images, "
          f"avg {len(out_df)/len(files):.1f} filaments/image", flush=True)


if __name__ == "__main__":
    main()
