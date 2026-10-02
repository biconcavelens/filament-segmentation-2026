"""Experiment A submission: Mask R-CNN(cls) + YOLO11m(cls=1.5)@1280 + the same
YOLO@2048 (native res), pooled via true-NMS dedup -- the 4-way ensemble
minus RT-DETR. Local val PQ 0.4458 in-sample / 0.4395 2-fold cross-fit, vs
the 2-way baseline's 0.4407 / 0.4354 (real 0.39) under the same methodology.

Reuses sweep_ensemble_4way.py's candidate/refine/dedup functions verbatim,
and fits the isotonic calibrators from its cached val candidates (all 116
images) instead of re-running val inference. accept/dedup are the peak
shared by both the in-sample and cross-fit sweeps.

Isotonic calibration depends only on the raw score, so candidates below
ACCEPT are dropped before the (expensive) refiner -- same result as the
sweep's refine-then-filter order, much cheaper on CPU.

    python predict_ensemble_expA.py               # test set -> submission_expA.csv (resumable)
    python predict_ensemble_expA.py --check-val 5 # live pipeline vs cached path on val
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from ultralytics import YOLO

from dataset import train_val_split, IMG_DIR
from train import build_from_checkpoint
from train_refiner import RefinerUNet
from predict_trained import to_rle
from sweep_ensemble_4way import (
    CACHE_PATH, MASKRCNN_CKPT, YOLO_CKPT, REFINER_CKPT, FLOOR_A, FLOOR_B1280, FLOOR_B2048,
    refine_candidate, dedup_nms_rle, paint_panoptic_rle, pq_against_gt,
    _yolo_style_candidates, fit_calibrators,
)

SOURCES = ["A", "B1280", "B2048"]
DEDUP_IOU = 0.05
ACCEPT = 0.5
TEST_DIR = Path("data/MAGFiLO_1.0_Kaggle_2026/test/test_images")
OUT = "submission_expA.csv"


def load_models(device):
    stateA = torch.load(MASKRCNN_CKPT, map_location=device)
    detA = build_from_checkpoint(stateA, num_classes=2).to(device)
    detA.load_state_dict(stateA["model"])
    detA.roi_heads.nms_thresh = 0.30
    detA.eval()
    rstate = torch.load(REFINER_CKPT, map_location=device)
    refiner = RefinerUNet(in_channels=rstate.get("in_channels", 1),
                          out_channels=rstate.get("out_channels", 1)).to(device)
    refiner.load_state_dict(rstate["model"])
    refiner.crop_size = rstate.get("crop_size", 256)
    refiner.eval()
    return detA, YOLO(YOLO_CKPT), refiner


@torch.no_grad()
def predict_one(detA, yolo, refiner, cals, device, img_path):
    gray = np.array(Image.open(img_path).convert("L"))
    rgb = np.array(Image.open(img_path).convert("RGB"))
    img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    outA = detA([img_t.to(device)])[0]
    scoresA = outA["scores"].cpu().numpy()
    masksA = outA["masks"].cpu().numpy()
    cands = {
        "A": [(float(scoresA[j]), (masksA[j, 0] > 0.5).astype(np.uint8))
              for j in range(len(scoresA))
              if scoresA[j] >= FLOOR_A and (masksA[j, 0] > 0.5).sum() > 0],
        "B1280": _yolo_style_candidates(yolo, img_path, 1280, FLOOR_B1280),
        "B2048": _yolo_style_candidates(yolo, img_path, 2048, FLOOR_B2048),
    }
    pooled = []
    for key in SOURCES:
        if not cands[key]:
            continue
        cal_scores = cals[key].predict([s for s, _ in cands[key]])
        for c, (_, coarse) in zip(cal_scores, cands[key]):
            if c < ACCEPT:
                continue
            ref = refine_candidate(refiner, device, gray, coarse)
            if ref.sum() > 0:
                pooled.append((float(c), to_rle(ref)))
    return paint_panoptic_rle(dedup_nms_rle(pooled, DEDUP_IOU))


def check_val(detA, yolo, refiner, cals, device, cache, n):
    """Live pipeline vs the sweep's cached path on the first n val images.
    Not bit-identical when the cache came from a different device (float
    drift in detector scores), but PQ and mask counts should be very close."""
    _, val_entries, _ = train_val_split(val_frac=0.1, seed=0)
    for i in range(n):
        per_source, gt = cache[i]
        pooled = []
        for key in SOURCES:
            refined = per_source[key]
            if refined:
                cs = cals[key].predict([s for s, _, _ in refined])
                pooled.extend((float(c), r) for c, (_, r, _) in zip(cs, refined) if c >= ACCEPT)
        cached_kept = paint_panoptic_rle(dedup_nms_rle(pooled, DEDUP_IOU))
        live_kept = predict_one(detA, yolo, refiner, cals, device, IMG_DIR / val_entries[i]["file_name"])
        c_num, c_den, c_tp = pq_against_gt(cached_kept, gt)
        l_num, l_den, l_tp = pq_against_gt(live_kept, gt)
        print(f"val[{i}] cached: n={len(cached_kept)} TP={c_tp} PQ={c_num / max(c_den, 1e-9):.3f} | "
              f"live: n={len(live_kept)} TP={l_tp} PQ={l_num / max(l_den, 1e-9):.3f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--check-val", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    cals = fit_calibrators(cache, SOURCES, list(range(len(cache))))
    detA, yolo, refiner = load_models(device)
    print(f"device={device} sources={SOURCES} dedup={DEDUP_IOU} accept={ACCEPT}", flush=True)

    if args.check_val:
        check_val(detA, yolo, refiner, cals, device, cache, args.check_val)
        return
    del cache

    partial, done_list = Path(OUT + ".partial"), Path(OUT + ".done")
    rows = pd.read_csv(partial, dtype=str).to_dict("records") if partial.exists() else []
    done = set(done_list.read_text().split()) if done_list.exists() else set()
    rows = [r for r in rows if r["filament_id"].rsplit("_", 1)[0] in done]
    files = sorted(TEST_DIR.iterdir())
    for path in files:
        if path.stem in done:
            continue
        kept = predict_one(detA, yolo, refiner, cals, device, path)
        rows.extend({"filament_id": f"{path.stem}_{k}", "segmentation_rle": r}
                    for k, r in enumerate(kept, 1))
        # rows first, then the done-marker; on resume, rows without a marker are dropped
        pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(partial, index=False)
        with open(done_list, "a") as f:
            f.write(path.stem + "\n")
        done.add(path.stem)
        print(f"{len(done)}/{len(files)} {path.stem}: {len(kept)} kept", flush=True)

    partial.replace(OUT)
    print(f"wrote {OUT}: {len(rows)} rows, {len(files)} images, "
          f"avg {len(rows) / len(files):.1f}/image", flush=True)


if __name__ == "__main__":
    main()
