"""Two-stage inference: Mask R-CNN detector -> fine-tuned SAM decoder.

Boxes come from the trained detector; SAM's frozen encoder + fine-tuned
decoder refines each box into a full-resolution mask.

Usage:
    python predict_sam.py --out submission.csv     # full test set
    python predict_sam.py --eval-val               # local PQ on val split
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import SamModel, SamProcessor
import pycocotools.mask as mu

from train import build_model as build_detector
from predict_trained import paint_panoptic, to_rle

D = Path("data/MAGFiLO_1.0_Kaggle_2026")
TEST_DIR = D / "test" / "test_images"
H, W = 2048, 2048

DETECTOR_SCORE_THRESHOLD = 0.80
DETECTOR_MASK_THRESHOLD = 0.5
SAM_MASK_THRESHOLD = 0.5
MIN_AREA = 20
CHUNK_SIZE = 20
COOLDOWN_SECONDS = 15


def load_models(detector_ckpt: str, sam_decoder_ckpt: str, device):
    detector = build_detector(num_classes=2).to(device)
    detector.load_state_dict(torch.load(detector_ckpt, map_location=device)["model"])
    detector.eval()

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    sam = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    state = torch.load(sam_decoder_ckpt, map_location=device)
    sam.mask_decoder.load_state_dict(state["mask_decoder"])
    sam.eval()
    return detector, sam, processor


def boxes_from_detector(detector, device, img_t: torch.Tensor):
    with torch.no_grad():
        out = detector([img_t.to(device)])[0]
    scores = out["scores"].cpu().numpy()
    masks = out["masks"].cpu().numpy()

    boxes, box_scores = [], []
    for j in range(len(scores)):
        if scores[j] < DETECTOR_SCORE_THRESHOLD:
            continue
        m = (masks[j, 0] > DETECTOR_MASK_THRESHOLD)
        if m.sum() == 0:
            continue
        ys, xs = np.where(m)
        boxes.append([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])
        box_scores.append(float(scores[j]))
    return boxes, box_scores


@torch.no_grad()
def refine_with_sam(sam, processor, device, pil_img: Image.Image, boxes: list):
    inputs = processor(pil_img, input_boxes=[boxes], return_tensors="pt").to(device)
    out = sam(**inputs, multimask_output=False)
    # upscale each low-res mask back to the *original* image resolution
    masks = processor.image_processor.post_process_masks(
        out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )[0]  # [N,1,H,W] bool/float per SAM postprocess (returns binary already thresholded at 0.0 logit by default)
    return masks.squeeze(1).numpy()  # [N,H,W]


@torch.no_grad()
def predict_one(detector, sam, processor, device, pil_img: Image.Image, img_t: torch.Tensor):
    boxes, box_scores = boxes_from_detector(detector, device, img_t)
    if not boxes:
        return []

    refined = refine_with_sam(sam, processor, device, pil_img, boxes)
    candidates = [(box_scores[j], (refined[j] > 0).astype(np.uint8)) for j in range(len(boxes))
                  if refined[j].sum() > 0]
    return paint_panoptic(candidates, MIN_AREA)


def predict(detector_ckpt: str, sam_ckpt: str, out_csv: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector, sam, processor = load_models(detector_ckpt, sam_ckpt, device)
    print(f"loaded detector={detector_ckpt} sam_decoder={sam_ckpt} on {device}")

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
            pil_img = Image.open(path).convert("RGB")
            img_t = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
            kept = predict_one(detector, sam, processor, device, pil_img, img_t)
            stem = path.stem
            rows.extend({"filament_id": f"{stem}_{k}", "segmentation_rle": to_rle(m)}
                        for k, m in enumerate(kept, 1))
            print(f"{stem}: {len(kept)} kept")

        pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(
            partial_path, index=False)
        done = min(chunk_start + CHUNK_SIZE, len(remaining))
        print(f"--- chunk done: {done}/{len(remaining)} remaining "
              f"({len(files) - len(remaining) + done}/{len(files)} total) ---")
        if done < len(remaining):
            torch.cuda.empty_cache() if device.type == "cuda" else None
            time.sleep(COOLDOWN_SECONDS)

    partial_path.replace(out_csv)
    out_df = pd.read_csv(out_csv)
    print(f"\nwrote {out_csv}: {len(out_df)} rows, {len(files)} images, "
          f"avg {len(out_df)/len(files):.1f} filaments/image")


def eval_val(detector_ckpt: str, sam_ckpt: str, n: int = 30):
    from dataset import train_val_split, IMG_DIR
    from pq import compute_pq

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector, sam, processor = load_models(detector_ckpt, sam_ckpt, device)
    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)

    pqs = []
    for i, e in enumerate(val_entries[:n], 1):
        pil_img = Image.open(IMG_DIR / e["file_name"]).convert("RGB")
        img_t = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
        kept = predict_one(detector, sam, processor, device, pil_img, img_t)
        pred_rles = [to_rle(m) for m in kept]

        gt_rles = []
        for a in per_image.get(e["id"], []):
            rles = mu.frPyObjects(a["segmentation"], H, W)
            gt_rles.append(to_rle(mu.decode(mu.merge(rles))))

        pq = compute_pq(pred_rles, gt_rles, H, W)
        pqs.append(pq)
        print(f"[{i}/{n}] {e['file_name']}: pred={len(kept):3d} gt={len(gt_rles):3d} PQ={pq:.3f}")
        if i % CHUNK_SIZE == 0 and i < n:
            torch.cuda.empty_cache()
            time.sleep(COOLDOWN_SECONDS)

    print(f"\nmean PQ over {len(pqs)} val images: {np.mean(pqs):.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--detector", default="checkpoints/maskrcnn_epoch3.pt")
    p.add_argument("--sam-decoder", default="checkpoints/sam_decoder_best.pt")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--eval-val", action="store_true")
    args = p.parse_args()

    if args.eval_val:
        eval_val(args.detector, args.sam_decoder)
    else:
        predict(args.detector, args.sam_decoder, args.out)
