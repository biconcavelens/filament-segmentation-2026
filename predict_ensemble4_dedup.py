"""Full-test-set prediction: four-source ensemble (Mask R-CNN + YOLO@1280 +
YOLO@2048 native-res + RT-DETR-cls2.0), all pooled via true cross-source
NMS dedup. Combines two local-only wins (RT-DETR recalibration, multi-scale
YOLO TTA) that each individually tied the real leaderboard rather than
beating it -- this tests whether combining them crosses the threshold
neither did alone. Local val PQ 0.4407 (2-way) -> 0.4465 (4-way), the best
local number of the whole session (sweep_ensemble_4way.py).
"""
import numpy as np
import pandas as pd
import torch
import pycocotools.mask as mu
from PIL import Image
from ultralytics import YOLO, RTDETR
from sklearn.isotonic import IsotonicRegression

from dataset import train_val_split, IMG_DIR, H, W
from train import build_from_checkpoint
from train_refiner import RefinerUNet
from crop_dataset import square_bounds, CROP_SIZE
from predict_trained import to_rle
from predict_refined import refine_with_tta

MASKRCNN_CKPT = "kaggle_kernel_maskrcnn_cls/output/checkpoints/maskrcnn_cls_epoch5.pt"
YOLO_CKPT = "kaggle_dataset_upload/yolo11m_cls_best.pt"
RTDETR_CKPT = "kaggle_kernel_rtdetr_cls/output/rtdetr_cls_best.pt"
REFINER_CKPT = "checkpoints/refiner_v5_best.pt"
FLOOR_A, FLOOR_B1280, FLOOR_B2048, FLOOR_C = 0.3, 0.05, 0.05, 0.05
MIN_AREA = 20
DEDUP_IOU = 0.05
ACCEPT_THRESH = 0.55

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


def _yolo_style_candidates(model, img_path, imgsz, floor):
    out = model.predict(source=str(img_path), imgsz=imgsz, conf=floor, verbose=False)[0]
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


def fit_calibration(detA, yolo, rtdetr, refiner, device, val_entries, per_image):
    scores_all = {"A": [], "B1280": [], "B2048": [], "C": []}
    labels_all = {"A": [], "B1280": [], "B2048": [], "C": []}
    with torch.no_grad():
        for i, e in enumerate(val_entries, 1):
            img_path = IMG_DIR / e["file_name"]
            gray = np.array(Image.open(img_path).convert("L"))
            rgb = np.array(Image.open(img_path).convert("RGB"))
            img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

            outA = detA([img_t.to(device)])[0]
            scoresA = outA["scores"].cpu().numpy()
            masksA = outA["masks"].cpu().numpy()
            candsA = [(float(scoresA[j]), (masksA[j, 0] > 0.5).astype(np.uint8))
                      for j in range(len(scoresA))
                      if scoresA[j] >= FLOOR_A and (masksA[j, 0] > 0.5).sum() > 0]

            candsB1280 = _yolo_style_candidates(yolo, img_path, 1280, FLOOR_B1280)
            candsB2048 = _yolo_style_candidates(yolo, img_path, 2048, FLOOR_B2048)
            candsC = _yolo_style_candidates(rtdetr, img_path, 1280, FLOOR_C)

            gt = []
            for a in per_image.get(e["id"], []):
                rles = mu.frPyObjects(a["segmentation"], H, W)
                gt.append({"size": [H, W], "counts": to_rle(mu.decode(mu.merge(rles))).encode()})

            for key, cands in [("A", candsA), ("B1280", candsB1280),
                                ("B2048", candsB2048), ("C", candsC)]:
                for score, coarse in cands:
                    ref = refine_candidate(refiner, device, gray, coarse)
                    label = 1 if best_iou_against_gt(ref, gt) > 0.5 else 0
                    scores_all[key].append(score)
                    labels_all[key].append(label)
            if i % 20 == 0:
                print(f"  calibration fit: {i}/{len(val_entries)}", flush=True)

    cals = {}
    for key in scores_all:
        cal = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        cal.fit(scores_all[key], labels_all[key])
        cals[key] = cal
    return cals


@torch.no_grad()
def predict_one(detA, yolo, rtdetr, refiner, cals, device, gray, rgb, img_path):
    img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    outA = detA([img_t.to(device)])[0]
    scoresA = outA["scores"].cpu().numpy()
    masksA = outA["masks"].cpu().numpy()
    candsA = [(float(scoresA[j]), (masksA[j, 0] > 0.5).astype(np.uint8))
              for j in range(len(scoresA))
              if scoresA[j] >= FLOOR_A and (masksA[j, 0] > 0.5).sum() > 0]

    candsB1280 = _yolo_style_candidates(yolo, img_path, 1280, FLOOR_B1280)
    candsB2048 = _yolo_style_candidates(yolo, img_path, 2048, FLOOR_B2048)
    candsC = _yolo_style_candidates(rtdetr, img_path, 1280, FLOOR_C)

    pooled = []
    for key, cands in [("A", candsA), ("B1280", candsB1280),
                        ("B2048", candsB2048), ("C", candsC)]:
        for score, coarse in cands:
            ref = refine_candidate(refiner, device, gray, coarse)
            if ref.sum() == 0:
                continue
            cal_score = float(cals[key].predict([score])[0])
            pooled.append((cal_score, ref))

    filtered = [(s, m) for s, m in pooled if s >= ACCEPT_THRESH]
    deduped = dedup_nms(filtered, DEDUP_IOU)
    return paint_panoptic(deduped)


def main():
    device = torch.device("cuda")
    stateA = torch.load(MASKRCNN_CKPT, map_location=device)
    detA = build_from_checkpoint(stateA, num_classes=2).to(device)
    detA.load_state_dict(stateA["model"])
    detA.roi_heads.nms_thresh = 0.30
    detA.eval()

    yolo = YOLO(YOLO_CKPT)
    rtdetr = RTDETR(RTDETR_CKPT)

    rstate = torch.load(REFINER_CKPT, map_location=device)
    refiner = RefinerUNet(in_channels=rstate.get("in_channels", 1),
                           out_channels=rstate.get("out_channels", 1)).to(device)
    refiner.load_state_dict(rstate["model"])
    refiner.eval()

    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)
    print("fitting isotonic calibration on held-out val split...", flush=True)
    cals = fit_calibration(detA, yolo, rtdetr, refiner, device, val_entries, per_image)
    print("calibration fit done.\n", flush=True)

    files = sorted(TEST_DIR.iterdir())
    rows = []
    for i, path in enumerate(files, 1):
        gray = np.array(Image.open(path).convert("L"))
        rgb = np.array(Image.open(path).convert("RGB"))
        kept = predict_one(detA, yolo, rtdetr, refiner, cals, device, gray, rgb, path)
        stem = path.stem
        rows.extend({"filament_id": f"{stem}_{k}", "segmentation_rle": to_rle(m)}
                    for k, m in enumerate(kept, 1))
        if i % 20 == 0:
            print(f"{i}/{len(files)} test images done", flush=True)

    out_df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    out_df.to_csv("submission_ensemble4_dedup.csv", index=False)
    print(f"\nwrote submission_ensemble4_dedup.csv: {len(out_df)} rows, {len(files)} images, "
          f"avg {len(out_df)/len(files):.1f} filaments/image", flush=True)


if __name__ == "__main__":
    main()
