"""[yolo11l variant: YOLO11l-seg at 1280 -- bigger backbone at its own inference res;
11s->11m was a real gain (0.4114->0.4171 local). Knobs below differ from the 2048 kernel.]

YOLO11m-seg (cls=1.5, the deployed recipe) trained at native 2048px, then
refined candidates for the val and test sets in the local cache format.

Why: YOLO@2048 *inference* on the 1280-trained model helped val every time
and scored 0.38 on all three submissions using it, while Mask R-CNN
*trained* at 2048 held 0.39 -- pointing at the train/inference resolution
mismatch. A local 2048 run on the 8GB laptop GPU spilled into shared memory
(epochs 10-134 min); a 16GB Kaggle GPU doesn't.

Training uses ultralytics' `time=` budget so the whole schedule (LR decay
included) fits in TRAIN_HOURS and the kernel completes -- a previous kernel
was cancelled at the ~12h session limit and, cancelled, kept no output.

Outputs (/kaggle/working), with TAG / SRC_KEY from the knobs below:
  {TAG}_best.pt, results.csv
  {TAG}_val_{SRC_KEY}.pkl   [(per_source {SRC_KEY: [(score, rle, tp_label)]}, gt_rles)] in val order
  {TAG}_test_{SRC_KEY}.pkl  [(per_source {SRC_KEY: [(score, rle, 0)]}, stem)] in sorted test order
Merge locally with merge_candidate_caches.py. (Version 1 of the 2048 kernel,
pushed before these knobs existed, names them yolo2048_{val,test}_B2048.pkl.)
"""
import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)

import json
import pickle
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pycocotools.mask as mu
import torch
import torch.nn as nn
from PIL import Image
from ultralytics import YOLO

DATA = Path("/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
IMG_DIR = DATA / "train" / "train_images"
TEST_DIR = DATA / "test" / "test_images"
ANN_PATH = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
YOLO_ROOT = Path("/kaggle/working/yolo_data")
WORK = Path("/kaggle/working")
RUN_DIR = WORK / "runs" / "train"

H, W = 2048, 2048
# per-kernel knobs (the yolo11l kernel is a copy with these changed)
MODEL_NAME = "yolo11l-seg.pt"
IMGSZ = 1280
SRC_KEY = "B1280"          # candidate-cache source this model replaces
TAG = "yolo11l_cls_1280"
BATCHES = (4, 2)           # tried in order; a smaller one only if the larger fails before a checkpoint
TRAIN_HOURS = 9.0          # kernel limit ~12h; leaves time for setup + candidate generation
CLS_GAIN = 1.5
FLOOR = 0.05               # same candidate floor as the local B2048 source
CROP_SIZE, CONTEXT, MIN_CROP = 256, 1.8, 96
REFINER_THRESHOLD, TTA_AREA_RATIO_MIN, TTA_AREA_RATIO_MAX = 0.5, 0.60, 1.70


# ---------- data (same split as dataset.train_val_split) ----------

def train_val_split(val_frac=0.1, seed=0):
    with open(ANN_PATH, encoding="utf-8") as f:
        coco = json.load(f)
    per_image = {}
    for a in coco["annotations"]:
        per_image.setdefault(a["image_id"], []).append(a)
    images = coco["images"]
    files = sorted(set(i["file_name"] for i in images))
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * val_frac))
    val_files = set(files[:n_val])
    train_entries = [i for i in images if i["file_name"] not in val_files]
    val_entries = [i for i in images if i["file_name"] in val_files]
    return train_entries, val_entries, per_image


def build_split(entries, per_image, split):
    img_dir, lbl_dir = YOLO_ROOT / "images" / split, YOLO_ROOT / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for e in entries:
        name = str(e["id"]).replace("/", "_").replace("\\", "_")
        dst = img_dir / f"{name}.jpeg"
        if not dst.exists():
            shutil.copy(IMG_DIR / e["file_name"], dst)
        lines = []
        for a in per_image.get(e["id"], []):
            for pts in a["segmentation"]:
                if len(pts) < 6:
                    continue
                lines.append("0 " + " ".join(f"{pts[i] / W:.6f} {pts[i + 1] / H:.6f}"
                                             for i in range(0, len(pts), 2)))
        (lbl_dir / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")


def build_dataset(train_entries, val_entries, per_image):
    build_split(train_entries, per_image, "train")
    build_split(val_entries, per_image, "val")
    yaml = YOLO_ROOT / "data.yaml"
    yaml.write_text(f"path: {YOLO_ROOT.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n",
                    encoding="utf-8")
    return yaml


# ---------- training ----------

def train(data_yaml):
    last = RUN_DIR / "weights" / "last.pt"
    for batch in BATCHES:
        try:
            if last.exists():
                print(f"resuming from {last}", flush=True)
                YOLO(str(last)).train(resume=True)
            else:
                print(f"training imgsz={IMGSZ} batch={batch} time={TRAIN_HOURS}h", flush=True)
                YOLO(MODEL_NAME).train(
                    data=str(data_yaml), epochs=40, time=TRAIN_HOURS, batch=batch, imgsz=IMGSZ,
                    patience=15, seed=0, deterministic=True, cls=CLS_GAIN, amp=True,
                    project=str(RUN_DIR.parent), name=RUN_DIR.name, exist_ok=True)
            return
        except Exception as e:  # typically CUDA OOM at batch 2 before any checkpoint exists
            print(f"training failed at batch={batch}: {e}", flush=True)
            torch.cuda.empty_cache()
            if last.exists():
                continue  # next loop iteration resumes
    raise RuntimeError(f"training failed at every batch size in {BATCHES}")


# ---------- refiner + candidates (verified-equivalent inline copy, same as other kernels) ----------

class ConvBlock(nn.Sequential):
    def __init__(self, in_c, out_c):
        super().__init__(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
        )


class RefinerUNet(nn.Module):
    def __init__(self, features=32, in_channels=1, out_channels=1):
        super().__init__()
        f = features
        self.in_channels, self.out_channels = in_channels, out_channels
        self.enc1, self.enc2 = ConvBlock(in_channels, f), ConvBlock(f, f * 2)
        self.enc3, self.enc4 = ConvBlock(f * 2, f * 4), ConvBlock(f * 4, f * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(f * 8, f * 16)
        self.up4, self.dec4 = nn.ConvTranspose2d(f * 16, f * 8, 2, 2), ConvBlock(f * 16, f * 8)
        self.up3, self.dec3 = nn.ConvTranspose2d(f * 8, f * 4, 2, 2), ConvBlock(f * 8, f * 4)
        self.up2, self.dec2 = nn.ConvTranspose2d(f * 4, f * 2, 2, 2), ConvBlock(f * 4, f * 2)
        self.up1, self.dec1 = nn.ConvTranspose2d(f * 2, f, 2, 2), ConvBlock(f * 2, f)
        self.head = nn.Conv2d(f, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


def square_bounds(mask, context=CONTEXT, minimum=MIN_CROP):
    height, width = mask.shape
    ys, xs = np.where(mask)
    cx, cy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    side = int(np.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * context))
    side = min(max(side, minimum), min(height, width))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0:
        x1 -= x0; x0 = 0
    if y0 < 0:
        y1 -= y0; y0 = 0
    if x1 > width:
        x0 -= x1 - width; x1 = width
    if y1 > height:
        y0 -= y1 - height; y1 = height
    return x0, y0, x1, y1


@torch.no_grad()
def refine_with_tta(refiner, device, x):
    views = {"identity": x, "hflip": x[:, :, ::-1], "vflip": x[:, ::-1, :], "hvflip": x[:, ::-1, ::-1]}
    probs = {}
    for name, view in views.items():
        x_t = torch.from_numpy(np.ascontiguousarray(view)).float().unsqueeze(0)
        p = torch.sigmoid(refiner(x_t.to(device)))[0, 0].cpu().numpy()
        if name == "hflip":
            p = np.fliplr(p)
        elif name == "vflip":
            p = np.flipud(p)
        elif name == "hvflip":
            p = np.flipud(np.fliplr(p))
        probs[name] = p
    identity_area = (probs["identity"] > REFINER_THRESHOLD).sum()
    avg_prob = np.mean(list(probs.values()), axis=0)
    if identity_area == 0:
        return avg_prob
    ratio = (avg_prob > REFINER_THRESHOLD).sum() / identity_area
    if ratio < TTA_AREA_RATIO_MIN or ratio > TTA_AREA_RATIO_MAX:
        return probs["identity"]
    return avg_prob


def to_rle(mask):
    return mu.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("utf-8")


@torch.no_grad()
def refine_candidate(refiner, device, gray, coarse):
    x0, y0, x1, y1 = square_bounds(coarse.astype(bool))
    crop = np.array(Image.fromarray(gray[y0:y1, x0:x1]).resize(
        (CROP_SIZE, CROP_SIZE), Image.BILINEAR)).astype(np.float32) / 255.0
    prob = refine_with_tta(refiner, device, np.stack([crop]))
    side = y1 - y0
    prob_full = np.array(Image.fromarray((prob * 255).astype(np.uint8)).resize(
        (side, side), Image.BILINEAR)).astype(np.float32) / 255.0
    full = np.zeros((H, W), dtype=np.uint8)
    full[y0:y1, x0:x1] = (prob_full > 0.5).astype(np.uint8)
    return full


def yolo_candidates(model, img_path):
    out = model.predict(source=str(img_path), imgsz=IMGSZ, conf=FLOOR, verbose=False)[0]
    cands = []
    if out.boxes is not None and len(out.boxes) > 0:
        boxes, scores = out.boxes.xyxy.cpu().numpy(), out.boxes.conf.cpu().numpy()
        for j in range(len(boxes)):
            x0, y0, x1, y1 = boxes[j]
            x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
            if x1 <= x0 or y1 <= y0:
                continue
            coarse = np.zeros((H, W), dtype=np.uint8)
            coarse[y0:y1, x0:x1] = 1
            cands.append((float(scores[j]), coarse))
    return cands


def build_candidates(model, refiner, device, items, out_path, with_gt):
    """items: (img_path, key, anns) -- key is the gt list (val) or stem (test)."""
    cache = []
    for i, (img_path, stem, anns) in enumerate(items, 1):
        gray = np.array(Image.open(img_path).convert("L"))
        gt = []
        for a in anns or []:
            gt.append(to_rle(mu.decode(mu.merge(mu.frPyObjects(a["segmentation"], H, W)))))
        gt_d = [{"size": [H, W], "counts": r.encode()} for r in gt]
        refined = []
        for score, coarse in yolo_candidates(model, img_path):
            ref = refine_candidate(refiner, device, gray, coarse)
            if ref.sum() == 0:
                continue
            label = 0
            if gt_d:
                iou = mu.iou([{"size": [H, W], "counts": to_rle(ref).encode()}], gt_d, [0] * len(gt_d))
                label = int(iou.max() > 0.5)
            refined.append((score, to_rle(ref), label))
        cache.append(({SRC_KEY: refined}, gt if with_gt else stem))
        if i % 20 == 0:
            print(f"  {out_path.name}: {i}/{len(items)}", flush=True)
    with open(out_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"saved {out_path}", flush=True)


def main():
    t0 = time.time()
    device = torch.device("cuda")
    print(torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f}GB",
          flush=True)
    train_entries, val_entries, per_image = train_val_split()
    data_yaml = build_dataset(train_entries, val_entries, per_image)
    print(f"dataset ready ({time.time() - t0:.0f}s)", flush=True)

    train(data_yaml)
    best = RUN_DIR / "weights" / "best.pt"
    shutil.copy(best, WORK / f"{TAG}_best.pt")
    shutil.copy(RUN_DIR / "results.csv", WORK / "results.csv")
    print(f"training done ({(time.time() - t0) / 3600:.2f}h)", flush=True)

    refiner_path = next(Path("/kaggle/input").rglob("refiner_v5_best.pt"))
    rstate = torch.load(refiner_path, map_location=device)
    refiner = RefinerUNet(in_channels=rstate.get("in_channels", 1),
                          out_channels=rstate.get("out_channels", 1)).to(device)
    refiner.load_state_dict(rstate["model"])
    refiner.eval()
    model = YOLO(str(best))

    build_candidates(model, refiner, device,
                     [(IMG_DIR / e["file_name"], None, per_image.get(e["id"], [])) for e in val_entries],
                     WORK / f"{TAG}_val_{SRC_KEY}.pkl", with_gt=True)
    build_candidates(model, refiner, device,
                     [(p, p.stem, None) for p in sorted(TEST_DIR.iterdir())],
                     WORK / f"{TAG}_test_{SRC_KEY}.pkl", with_gt=False)
    print(f"all done ({(time.time() - t0) / 3600:.2f}h)", flush=True)


if __name__ == "__main__":
    main()
