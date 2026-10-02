"""Mask2Former (Swin-Tiny) with the backbone UNFROZEN, then val/test candidates
in the local cache format (source key "D") for the ensemble.

Our only transformer so far (Mask2Former, real 0.28 / later 0.345 local) kept
the whole Swin encoder frozen -- COCO features never adapted to H-alpha, only
the decoders trained. A competitor reports a transformer-based model at local
PQ 0.48 / public 0.41. Here the encoder trains too, at a 10x lower lr than the
decoders so the COCO features aren't wrecked early.

Training runs on a time budget with a time-based polynomial lr decay, so the
schedule completes however fast epochs turn out to be (a cancelled kernel
keeps no output). Best checkpoint by val loss is used for candidates.

Env overrides (for a local CPU smoke test): M2F_DATA, M2F_WORK, M2F_HOURS, M2F_SMOKE=1.

Outputs (WORK): m2f_unfrozen_best.pt, train log lines in stdout,
  m2f_val_D.pkl  [(per_source {"D": [(score, rle, tp_label)]}, gt_rles)] in val order
  m2f_test_D.pkl [(per_source {"D": [(score, rle, 0)]}, stem)] in sorted test order
"""
import json
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pycocotools.mask as mu
import torch
from PIL import Image
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

DATA = Path(os.environ.get("M2F_DATA", "/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"))
WORK = Path(os.environ.get("M2F_WORK", "/kaggle/working"))
WORK.mkdir(parents=True, exist_ok=True)
IMG_DIR = DATA / "train" / "train_images"
TEST_DIR = DATA / "test" / "test_images"
ANN_PATH = DATA / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
SMOKE = os.environ.get("M2F_SMOKE") == "1"

H, W = 2048, 2048
INPUT_SIZE = 1024
CHECKPOINT_NAME = "facebook/mask2former-swin-tiny-coco-instance"
TRAIN_HOURS = float(os.environ.get("M2F_HOURS", "8.0"))  # kernel limit ~12h; leaves time for candidates
LR_HEADS, LR_BACKBONE = 1e-4, 1e-5
FLOOR = 0.2  # candidate score floor; isotonic calibration downstream decides what is kept


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
    return ([i for i in images if i["file_name"] not in val_files],
            [i for i in images if i["file_name"] in val_files], per_image)


def to_rle(mask):
    return mu.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("utf-8")


def gt_masks(anns):
    out = []
    for a in anns:
        m = mu.decode(mu.merge(mu.frPyObjects(a["segmentation"], H, W)))
        if m.sum():
            out.append(m)
    return out


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, entries, per_image, processor, augment):
        self.entries = [e for e in entries if per_image.get(e["id"])]
        self.per_image, self.processor, self.augment = per_image, processor, augment

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        img = np.array(Image.open(IMG_DIR / e["file_name"]).convert("RGB"))
        masks = np.stack(gt_masks(self.per_image[e["id"]])).astype(np.float32)
        if self.augment:
            if random.random() < 0.5:
                img, masks = img[:, ::-1], masks[:, :, ::-1]
            if random.random() < 0.5:
                img, masks = img[::-1], masks[:, ::-1]
        pv = self.processor(Image.fromarray(np.ascontiguousarray(img)), return_tensors="pt")["pixel_values"][0]
        mt = torch.nn.functional.interpolate(torch.from_numpy(np.ascontiguousarray(masks))[None],
                                             size=(INPUT_SIZE, INPUT_SIZE), mode="nearest")[0]
        return pv, mt, torch.zeros(len(masks), dtype=torch.long)


def loss_on(model, device, batch):
    pv, mt, cl = batch
    return model(pixel_values=pv[None].to(device), mask_labels=[mt.to(device)], class_labels=[cl.to(device)]).loss


@torch.no_grad()
def candidates(model, processor, device, items, with_gt):
    """items: (path, stem, anns). One entry per image, in the given order."""
    cache = []
    for i, (path, stem, anns) in enumerate(items, 1):
        inputs = processor(Image.open(path).convert("RGB"), return_tensors="pt").to(device)
        out = model(**inputs)
        res = processor.post_process_instance_segmentation(
            out, threshold=FLOOR, mask_threshold=0.5, target_sizes=[(H, W)], return_binary_maps=True)[0]
        segs, info = res["segmentation"], res["segments_info"]
        gt = [to_rle(m) for m in gt_masks(anns)] if with_gt else []
        gt_d = [{"size": [H, W], "counts": r.encode()} for r in gt]
        cands = []
        if segs is not None and segs.numel() and segs.ndim == 3:
            segs = segs.cpu().numpy().astype(np.uint8)
            for k, s in enumerate(info):
                if segs[k].sum() < 20:
                    continue
                rle = to_rle(segs[k])
                label = 0
                if gt_d:
                    label = int(mu.iou([{"size": [H, W], "counts": rle.encode()}], gt_d, [0] * len(gt_d)).max() > 0.5)
                cands.append((float(s["score"]), rle, label))
        cache.append(({"D": cands}, gt if with_gt else stem))
        if i % 20 == 0:
            print(f"  candidates {i}/{len(items)}", flush=True)
    return cache


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() else "cpu")
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0), flush=True)
    processor = Mask2FormerImageProcessor.from_pretrained(
        CHECKPOINT_NAME, size={"shortest_edge": INPUT_SIZE, "longest_edge": INPUT_SIZE + 32}, do_reduce_labels=False)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        CHECKPOINT_NAME, num_labels=1, ignore_mismatched_sizes=True).to(device)

    train_e, val_e, per_image = train_val_split()
    if SMOKE:
        train_e, val_e = train_e[:4], val_e[:2]
    train_ds = TrainSet(train_e, per_image, processor, augment=True)
    val_ds = TrainSet(val_e, per_image, processor, augment=False)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=lambda b: b[0],
                                         num_workers=0 if SMOKE else 2)

    enc = [p for n, p in model.named_parameters() if n.startswith("model.pixel_level_module.encoder")]
    enc_ids = {id(p) for p in enc}
    rest = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": enc, "lr": LR_BACKBONE}, {"params": rest, "lr": LR_HEADS}], weight_decay=0.05)
    base = [g["lr"] for g in opt.param_groups]
    budget = TRAIN_HOURS * 3600
    print(f"train {len(train_ds)} / val {len(val_ds)} images, encoder params {sum(p.numel() for p in enc) / 1e6:.1f}M "
          f"(trainable), budget {TRAIN_HOURS}h", flush=True)

    best, epoch = float("inf"), 0
    while time.time() - t0 < budget:
        model.train()
        te, run = time.time(), 0.0
        for i, batch in enumerate(loader):
            frac = min((time.time() - t0) / budget, 1.0)
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * max(0.02, (1 - frac) ** 0.9)
            loss = loss_on(model, device, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step()
            run += loss.item()
            if SMOKE and i >= 1:
                break
            if time.time() - t0 > budget:
                break
        model.eval()
        with torch.no_grad():
            val = float(np.mean([loss_on(model, device, val_ds[j]).item() for j in range(len(val_ds))]))
        print(f"epoch {epoch}: train_loss={run / max(i + 1, 1):.3f} val_loss={val:.3f} "
              f"({time.time() - te:.0f}s, {(time.time() - t0) / 3600:.2f}h elapsed)", flush=True)
        if val < best:
            best = val
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val}, WORK / "m2f_unfrozen_best.pt")
        epoch += 1
        if SMOKE:
            break

    model.load_state_dict(torch.load(WORK / "m2f_unfrozen_best.pt", map_location=device)["model"])
    model.eval()
    _, val_all, _ = train_val_split()
    test_items = [(p, p.stem, None) for p in sorted(TEST_DIR.iterdir())]
    val_items = [(IMG_DIR / e["file_name"], None, per_image.get(e["id"], [])) for e in val_all]
    if SMOKE:
        val_items, test_items = val_items[:2], test_items[:2]
    for name, items, with_gt in (("m2f_val_D.pkl", val_items, True), ("m2f_test_D.pkl", test_items, False)):
        cache = candidates(model, processor, device, items, with_gt)
        with open(WORK / name, "wb") as f:
            pickle.dump(cache, f)
        n = sum(len(ps["D"]) for ps, _ in cache)
        print(f"saved {name}: {len(cache)} images, {n} candidates", flush=True)
    print(f"all done ({(time.time() - t0) / 3600:.2f}h)", flush=True)


if __name__ == "__main__":
    main()
