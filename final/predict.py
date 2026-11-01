"""Final system (public leaderboard PQ 0.40): GONG H-alpha images -> submission CSV, from the weights in final/weights.

    python final/predict.py --images data/MAGFiLO_1.0_Kaggle_2026/test/test_images --out submission.csv

Per image:
  1. four detectors (Mask R-CNN @2048, YOLO11m-seg, RT-DETR-l, YOLO11l-seg @1280) on the image and on its mirror
  2. every proposal is redrawn by a crop-level U-Net refiner: 256 px and 512 px on the normal proposals, 256 px on
     the mirrored ones -> 3 refined versions per detector (12 sources)
  3. a ConvNeXt-Tiny U-Net segments the whole image; its components are split into instances (source S)
  4. isotonic calibration per source, then leaders (score >= 0.5, NMS IoU 0.05) take a score-weighted pixel vote of
     all overlapping masks; masks are painted without overlap.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from PIL import Image
from ultralytics import RTDETR, YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from predict_trained import to_rle  # noqa: E402
from sweep_ensemble_4way import (FLOOR_A, FLOOR_B1280, FLOOR_C, H, W, _yolo_style_candidates,  # noqa: E402
                                 paint_panoptic_rle, refine_candidate)
from sweep_mask_fusion import fuse_image  # noqa: E402
from sweep_semseg import instances  # noqa: E402
from train import build_from_checkpoint  # noqa: E402
from train_refiner import load_refiner  # noqa: E402

WEIGHTS = Path(__file__).resolve().parent / "weights"
FUSE = (0.5, 0.3, 0.3)  # fuse IoU, member score floor, vote share
LEADERS = ["Av", "B1280v", "Cv", "L1280v", "S"]
MIRROR = {"A": "AF", "B1280": "BF", "C": "CF", "L1280": "LF"}
SEM = dict(t_fg=0.5, t_sp=0.3, close=3, min_area=100)


class SemNet(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.net = smp.Unet(encoder, encoder_weights=None, in_channels=1, classes=2)

    def forward(self, x):
        return self.net((x - 0.45) / 0.225)


@torch.no_grad()
def semantic_probs(model, gray, device):
    """4-flip TTA probability maps (fg, spine), quantised to uint8 like the saved maps the system was tuned on."""
    x = torch.from_numpy(gray.astype(np.float32) / 255.0)[None, None].to(device)
    acc = 0
    for flip in [(), (3,), (2,), (2, 3)]:
        xi = torch.flip(x, flip) if flip else x
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            p = torch.sigmoid(model(xi).float())
        acc = acc + (torch.flip(p, flip) if flip else p)
    p = ((acc / 4)[0].cpu().numpy() * 255).round().astype(np.uint8)
    return p[0].astype(np.float32) / 255.0, p[1].astype(np.float32) / 255.0


def load_models(device):
    st = torch.load(WEIGHTS / "maskrcnn.pt", map_location="cpu")
    mrcnn = build_from_checkpoint(st, num_classes=2, pretrained=False)
    mrcnn.load_state_dict(st["model"])
    mrcnn.roi_heads.nms_thresh = 0.30
    st = torch.load(WEIGHTS / "semseg.pt", map_location="cpu")
    sem = SemNet(st["encoder"])
    sem.load_state_dict(st["model"])
    return dict(
        mrcnn=mrcnn.to(device).eval(), sem=sem.to(device).eval(),
        yolo_m=YOLO(str(WEIGHTS / "yolo11m.pt")), yolo_l=YOLO(str(WEIGHTS / "yolo11l.pt")),
        rtdetr=RTDETR(str(WEIGHTS / "rtdetr.pt")),
        ref256=load_refiner(str(WEIGHTS / "refiner256.pt"), device), ref512=load_refiner(str(WEIGHTS / "refiner512.pt"), device))


def detect(m, path, device, mirror):
    """Coarse proposals {A, B1280, C, L1280: [(score, mask)]} in the original frame."""
    rgb = np.array(Image.open(path).convert("RGB"))
    src = path
    if mirror:
        rgb = np.ascontiguousarray(rgb[:, ::-1])
        src = np.ascontiguousarray(rgb[:, :, ::-1])  # ultralytics takes BGR arrays
    img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        out = m["mrcnn"]([img_t.to(device)])[0]
    sc, mk = out["scores"].cpu().numpy(), out["masks"].cpu().numpy()
    raw = {"A": [(float(sc[j]), (mk[j, 0] > 0.5).astype(np.uint8)) for j in range(len(sc))
                 if sc[j] >= FLOOR_A and (mk[j, 0] > 0.5).sum() > 0],
           "B1280": _yolo_style_candidates(m["yolo_m"], src, 1280, FLOOR_B1280),
           "C": _yolo_style_candidates(m["rtdetr"], src, 1280, FLOOR_C),
           "L1280": _yolo_style_candidates(m["yolo_l"], src, 1280, FLOOR_B1280)}
    if mirror:
        raw = {k: [(s, np.ascontiguousarray(x[:, ::-1])) for s, x in v] for k, v in raw.items()}
    return raw


def refine(refiner, device, gray, raw, suffix):
    out = {}
    for key, cands in raw.items():
        keep = []
        for score, coarse in cands:
            ref = refine_candidate(refiner, device, gray, coarse)
            if ref.sum() > 0:
                keep.append((score, to_rle(ref)))
        out[key + suffix] = keep
    return out


def sources_for(m, path, device):
    gray = np.array(Image.open(path).convert("L"))
    normal, mirrored = detect(m, path, device, False), detect(m, path, device, True)
    src = {}
    src.update(refine(m["ref256"], device, gray, normal, "v"))  # Av, B1280v, Cv, L1280v
    src.update(refine(m["ref512"], device, gray, normal, ""))  # A, B1280, C, L1280
    src.update({MIRROR[k]: v for k, v in refine(m["ref256"], device, gray, mirrored, "").items()})
    fg, sp = semantic_probs(m["sem"], gray, device)
    src["S"] = [(s, to_rle(x)) for s, x in instances(fg, sp, SEM["t_fg"], SEM["t_sp"], SEM["close"], SEM["min_area"])]
    return src


def calibrate(cals, src):
    pooled = []
    for key, cands in src.items():
        if cands:
            c = cals[key]
            vals = np.interp([s for s, _ in cands], c["x"], c["y"])
            pooled += [(float(v), r, key) for v, (_, r) in zip(vals, cands)]
    return sorted(pooled, key=lambda t: -t[0])


def main():
    global WEIGHTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (smoke test)")
    ap.add_argument("--pause-every", type=int, default=0, help="rest the GPU after this many images (overheating laptops)")
    ap.add_argument("--pause-sec", type=int, default=60)
    ap.add_argument("--weights", default=str(WEIGHTS), help="weight folder (final/weights_all = trained on train + validation)")
    a = ap.parse_args()
    WEIGHTS = Path(a.weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cals = json.load(open(WEIGHTS / "calibrators.json"))
    models = load_models(device)
    paths = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    paths = paths[:a.limit] if a.limit else paths
    rows, done = [], set()
    done_file = Path(a.out + ".done")  # a crash costs one image: finished images are skipped on restart
    if done_file.exists() and Path(a.out).exists():
        try:
            rows = pd.read_csv(a.out).to_dict("records")
            done = set(done_file.read_text().split())
        except Exception:  # a crash mid-write can leave a zero-filled CSV: start over rather than trust it
            rows, done = [], set()
            done_file.unlink()
    for i, path in enumerate(paths, 1):
        if path.name in done:
            continue
        pooled = calibrate(cals, sources_for(models, path, device))
        kept = paint_panoptic_rle(fuse_image(pooled, [FUSE], LEADERS)[FUSE])
        rows += [{"filament_id": f"{path.stem}_{k}", "segmentation_rle": r} for k, r in enumerate(kept, 1)]
        print(f"{i}/{len(paths)} {path.name}: {len(kept)} filaments", flush=True)
        if a.pause_every and i % a.pause_every == 0:
            time.sleep(a.pause_sec)
        tmp = Path(a.out + ".tmp")  # write-then-rename: a crash never leaves a half-written CSV
        with open(tmp, "w", newline="") as f:
            pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(f, index=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, a.out)
        with open(done_file, "a") as f:
            f.write(path.name + "\n")
    print(f"wrote {a.out}: {len(rows)} rows, {len(paths)} images")


if __name__ == "__main__":
    main()
