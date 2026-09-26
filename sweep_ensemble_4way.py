"""Four-source ensemble: Mask R-CNN + YOLO@1280 + YOLO@2048 (native res) +
RT-DETR(cls=2.0/60ep), all pooled via the same true-NMS-dedup mechanism.

Two independent local wins this session did NOT transfer to the real
leaderboard on their own: RT-DETR recalibration (+0.0011 local, tied real)
and multi-scale YOLO TTA (+0.0199 local, largest of the session, but STILL
tied real at 0.38 solo). Every genuine real-score improvement so far has
come from ensembling multiple sources, not tuning any single one -- this
tests whether combining both non-transferring wins into the existing
validated ensemble crosses the leaderboard's rounding threshold where
neither did alone.

The GPU pass caches every refined candidate (score, RLE, TP label) per
source, so any subset can be re-evaluated on CPU afterwards:
    python sweep_ensemble_4way.py                                  # GPU pass + full 4-way sweep
    python sweep_ensemble_4way.py --from-cache --sources A B1280 B2048 --crossfit
--crossfit fits each source's isotonic calibrator on one half of the val
images and scores the other half (2-fold), instead of calibrating and
scoring on the same 116 images -- the suspected cause of the 4-way's
local-win/real-regression gap.
"""
import argparse
import os
import pickle
from pathlib import Path

import numpy as np
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
CACHE_PATH = "ensemble4_candidates.pkl"
TEST_CACHE_PATH = "ensemble4_test_candidates.pkl"
TEST_DIR = Path("data/MAGFiLO_1.0_Kaggle_2026/test/test_images")
ALL_SOURCES = ["A", "B1280", "B2048", "C"]


@torch.no_grad()
def refine_candidate(refiner, device, gray, coarse):
    x0, y0, x1, y1 = square_bounds(coarse.astype(bool))
    crop = np.array(Image.fromarray(gray[y0:y1, x0:x1]).resize(
        (CROP_SIZE, CROP_SIZE), Image.BILINEAR)).astype(np.float32) / 255.0
    channels = [crop]
    if refiner.in_channels == 2:  # hint refiner (v8): the proposal itself, same frame
        channels.append((np.array(Image.fromarray(coarse[y0:y1, x0:x1].astype(np.uint8) * 255).resize(
            (CROP_SIZE, CROP_SIZE), Image.NEAREST)) > 127).astype(np.float32))
    prob = refine_with_tta(refiner, device, np.stack(channels))
    side = y1 - y0
    prob_full = np.array(Image.fromarray((prob * 255).astype(np.uint8)).resize(
        (side, side), Image.BILINEAR)).astype(np.float32) / 255.0
    refined_crop = (prob_full > 0.5).astype(np.uint8)
    full_mask = np.zeros((H, W), dtype=np.uint8)
    full_mask[y0:y1, x0:x1] = refined_crop
    return full_mask


def best_iou_against_gt(mask, gt_rles):
    if not gt_rles:
        return 0.0
    pred_rle = {"size": [H, W], "counts": to_rle(mask).encode()}
    iou = mu.iou([pred_rle], gt_rles, [0] * len(gt_rles))
    return float(iou.max())


def dedup_nms_rle(candidates, iou_thresh):
    candidates = sorted(candidates, key=lambda x: -x[0])
    if not candidates:
        return []
    rles = [{"size": [H, W], "counts": r.encode()} for _, r in candidates]
    iou = mu.iou(rles, rles, [0] * len(rles))  # same IoU as dense masks, computed on RLE
    accepted = []
    for i in range(len(candidates)):
        if all(iou[i, j] <= iou_thresh for j in accepted):
            accepted.append(i)
    return [candidates[i] for i in accepted]


def paint_panoptic_rle(candidates, min_area=MIN_AREA):
    candidates = sorted(candidates, key=lambda x: -x[0])
    claimed = np.zeros((H, W), dtype=np.uint8)
    kept = []
    for score, rle in candidates:
        binary = mu.decode({"size": [H, W], "counts": rle.encode()})
        remaining = binary & (1 - claimed)
        area = int(remaining.sum())
        if area < min_area:
            continue
        claimed |= remaining
        kept.append(to_rle(remaining))
    return kept


def pq_against_gt(kept_rles, gt_rles):
    pred = [{"size": [H, W], "counts": r.encode()} for r in kept_rles]
    gt_dicts = [{"size": [H, W], "counts": r.encode()} for r in gt_rles]
    if pred and gt_dicts:
        iou = mu.iou(pred, gt_dicts, [0] * len(gt_dicts))
        best_per_gt = iou.max(axis=0)
    else:
        best_per_gt = np.zeros(len(gt_rles))
    m_tp = int((best_per_gt > 0.5).sum())
    n_fp = len(pred) - m_tp
    n_fn = len(gt_rles) - m_tp
    num = float(best_per_gt[best_per_gt > 0.5].sum())
    den = m_tp + 0.5 * n_fp + 0.5 * n_fn
    return num, den, m_tp


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


def build_cache(test=False, sources=ALL_SOURCES):
    """val: entries are (per_source, gt_rles), saved at the end.
    test: entries are (per_source, image_stem) with label=0, saved every 10
    images and resumed from the partial file -- the CPU test pass takes hours."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stateA = torch.load(MASKRCNN_CKPT, map_location=device)
    detA = build_from_checkpoint(stateA, num_classes=2).to(device)
    detA.load_state_dict(stateA["model"])
    detA.roi_heads.nms_thresh = 0.30
    detA.eval()

    yolo = YOLO(YOLO_CKPT)
    rtdetr = RTDETR(RTDETR_CKPT) if "C" in sources else None

    rstate = torch.load(REFINER_CKPT, map_location=device)
    refiner = RefinerUNet(in_channels=rstate.get("in_channels", 1),
                           out_channels=rstate.get("out_channels", 1)).to(device)
    refiner.load_state_dict(rstate["model"])
    refiner.eval()

    if test:
        out_path = TEST_CACHE_PATH
        items = [(p, p.stem, None) for p in sorted(TEST_DIR.iterdir())]
    else:
        out_path = CACHE_PATH
        _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)
        items = [(IMG_DIR / e["file_name"], None, per_image.get(e["id"], [])) for e in val_entries]

    per_image_cache = []
    if test and os.path.exists(out_path + ".partial"):
        with open(out_path + ".partial", "rb") as f:
            per_image_cache = pickle.load(f)
        print(f"resuming test cache at {len(per_image_cache)}/{len(items)}", flush=True)

    with torch.no_grad():
        for i, (img_path, stem, anns) in enumerate(items, 1):
            if i <= len(per_image_cache):
                continue
            gray = np.array(Image.open(img_path).convert("L"))
            rgb = np.array(Image.open(img_path).convert("RGB"))
            img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

            outA = detA([img_t.to(device)])[0]
            scoresA = outA["scores"].cpu().numpy()
            masksA = outA["masks"].cpu().numpy()
            candsA_raw = [(float(scoresA[j]), (masksA[j, 0] > 0.5).astype(np.uint8))
                          for j in range(len(scoresA))
                          if scoresA[j] >= FLOOR_A and (masksA[j, 0] > 0.5).sum() > 0]

            raw = {"A": candsA_raw}
            if "B1280" in sources:
                raw["B1280"] = _yolo_style_candidates(yolo, img_path, 1280, FLOOR_B1280)
            if "B2048" in sources:
                raw["B2048"] = _yolo_style_candidates(yolo, img_path, 2048, FLOOR_B2048)
            if rtdetr is not None:
                raw["C"] = _yolo_style_candidates(rtdetr, img_path, 1280, FLOOR_C)

            gt = []
            for a in anns or []:
                rles = mu.frPyObjects(a["segmentation"], H, W)
                gt.append(to_rle(mu.decode(mu.merge(rles))))
            gt_dicts = [{"size": [H, W], "counts": r.encode()} for r in gt]

            per_source = {}
            for key in sources:
                refined = []
                for score, coarse in raw[key]:
                    ref = refine_candidate(refiner, device, gray, coarse)
                    if ref.sum() == 0:
                        continue
                    label = 1 if best_iou_against_gt(ref, gt_dicts) > 0.5 else 0
                    refined.append((score, to_rle(ref), label))
                per_source[key] = refined

            per_image_cache.append((per_source, stem if test else gt))
            if i % 10 == 0:
                print(f"  cached {i}/{len(items)}", flush=True)
                if test:
                    with open(out_path + ".partial", "wb") as f:
                        pickle.dump(per_image_cache, f)

    with open(out_path, "wb") as f:
        pickle.dump(per_image_cache, f)
    print(f"saved {out_path}", flush=True)
    return per_image_cache


def fit_calibrators(per_image_cache, sources, image_idx):
    cals = {}
    for key in sources:
        scores = [s for i in image_idx for s, _, _ in per_image_cache[i][0][key]]
        labels = [l for i in image_idx for _, _, l in per_image_cache[i][0][key]]
        cal = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        cal.fit(scores, labels)
        cals[key] = cal
    return cals


def calibrate(per_image_cache, sources, crossfit):
    n = len(per_image_cache)
    if crossfit:
        # 2-fold: each image is scored by calibrators that never saw it
        folds = [(list(range(f, n, 2)), [i for i in range(n) if i % 2 != f]) for f in (0, 1)]
    else:
        folds = [(list(range(n)), list(range(n)))]
    pooled_by_image = [None] * n
    for score_idx, fit_idx in folds:
        cals = fit_calibrators(per_image_cache, sources, fit_idx)
        for i in score_idx:
            per_source, gt = per_image_cache[i]
            pooled = []
            for key in sources:
                refined = per_source[key]
                if refined:
                    cs = cals[key].predict([s for s, _, _ in refined])
                    pooled.extend((float(c), r) for c, (_, r, _) in zip(cs, refined))
            pooled_by_image[i] = (pooled, gt)
    return pooled_by_image


def main():
    global REFINER_CKPT, CACHE_PATH, TEST_CACHE_PATH, MASKRCNN_CKPT, YOLO_CKPT
    p = argparse.ArgumentParser()
    p.add_argument("--from-cache", action="store_true")
    p.add_argument("--sources", nargs="+", default=ALL_SOURCES,
                   help="cache source keys; building only knows ALL_SOURCES, merged caches may add more")
    p.add_argument("--crossfit", action="store_true")
    p.add_argument("--build-test-cache", action="store_true")
    p.add_argument("--refiner", default=REFINER_CKPT)
    p.add_argument("--maskrcnn", default=MASKRCNN_CKPT)
    p.add_argument("--yolo", default=YOLO_CKPT, help="YOLO weights for the B1280/B2048 sources")
    p.add_argument("--cache", default=CACHE_PATH, help="val candidate cache path")
    p.add_argument("--test-cache", default=TEST_CACHE_PATH)
    args = p.parse_args()
    REFINER_CKPT, CACHE_PATH, TEST_CACHE_PATH = args.refiner, args.cache, args.test_cache
    MASKRCNN_CKPT, YOLO_CKPT = args.maskrcnn, args.yolo

    if args.build_test_cache:
        build_cache(test=True, sources=args.sources)
        return
    if args.from_cache:
        with open(CACHE_PATH, "rb") as f:
            per_image_cache = pickle.load(f)
    else:
        per_image_cache = build_cache(sources=args.sources)

    for key in args.sources:
        cands = [c for per_source, _ in per_image_cache for c in per_source[key]]
        print(f"{key}: {len(cands)} candidates ({sum(l for _, _, l in cands)} TP)", flush=True)
    print(f"sources={args.sources} crossfit={args.crossfit}", flush=True)

    calibrated_cache = calibrate(per_image_cache, args.sources, args.crossfit)
    del per_image_cache

    def pq_for(dedup_iou, accept_thresh):
        pq_num = pq_den = 0.0
        tp = 0
        for pooled, gt in calibrated_cache:
            filtered = [(s, r) for s, r in pooled if s >= accept_thresh]
            deduped = dedup_nms_rle(filtered, dedup_iou)
            kept = paint_panoptic_rle(deduped)
            num, den, m_tp = pq_against_gt(kept, gt)
            pq_num += num
            pq_den += den
            tp += m_tp
        return (pq_num / pq_den if pq_den else 0.0), tp

    print("\n=== references: 2-way ens 0.4407 (real 0.39), 3-way+RTDETR 0.4418 (real 0.39, tied)")
    print("=== solo multiscale YOLO 0.4451 (real 0.38, tied) ===\n")
    results = []
    for dedup_iou in [0.03, 0.05, 0.08, 0.1]:
        for accept_thresh in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
            pq, tp = pq_for(dedup_iou, accept_thresh)
            print(f"dedup_iou={dedup_iou} accept={accept_thresh}: PQ={pq:.4f} TP={tp}", flush=True)
            results.append((pq, dedup_iou, accept_thresh, tp))

    results.sort(reverse=True)
    print("\n=== top 5 ===")
    for pq, dedup_iou, accept_thresh, tp in results[:5]:
        print(f"PQ={pq:.4f}  dedup_iou={dedup_iou}  accept={accept_thresh}  TP={tp}")


if __name__ == "__main__":
    main()
