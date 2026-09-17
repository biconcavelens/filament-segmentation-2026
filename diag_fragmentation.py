"""Measure one-to-many (fragmentation) and many-to-one (over-merging) rates
on our current best config -- a rubric-scored criterion ("distribution of
one-to-many and many-to-one relations") that every experiment this session
has ignored in favor of optimizing PQ/TP/FP directly. Runs locally (GPU is
free now) since it's cheap: one inference pass over the 116-image val split.
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

YOLO_CKPT = "kaggle_kernel_train_m/output/yolo11m_best.pt"
REFINER_CKPT = "checkpoints/refiner_v5_best.pt"
YOLO_CONF = 0.35
MIN_AREA = 20
OVERLAP_THRESH = 0.10  # "significant" overlap for fragmentation/merge counting


@torch.no_grad()
def predict_one(yolo, refiner, device, gray, img_path):
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

    candidates.sort(key=lambda x: -x[0])
    claimed = np.zeros((H, W), dtype=np.uint8)
    kept = []
    for score, binary in candidates:
        remaining = binary & (1 - claimed)
        if int(remaining.sum()) < MIN_AREA:
            continue
        claimed |= remaining
        kept.append(remaining)
    return kept


def main():
    device = torch.device("cuda")
    yolo = YOLO(YOLO_CKPT)
    rstate = torch.load(REFINER_CKPT, map_location=device)
    refiner = RefinerUNet(in_channels=rstate.get("in_channels", 1),
                           out_channels=rstate.get("out_channels", 1)).to(device)
    refiner.load_state_dict(rstate["model"])
    refiner.eval()

    _, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)

    n_gt_total = n_pred_total = 0
    n_fragmented = 0    # GT instances matched by >1 prediction at OVERLAP_THRESH
    n_merged_preds = 0  # predictions matched by >1 GT instance at OVERLAP_THRESH
    n_gt_involved_in_merge = 0
    frag_examples = []
    merge_examples = []

    pq_num = pq_den = 0.0
    tp = 0

    for i, e in enumerate(val_entries, 1):
        img_path = IMG_DIR / e["file_name"]
        gray = np.array(Image.open(img_path).convert("L"))
        kept = predict_one(yolo, refiner, device, gray, img_path)
        pred = [{"size": [H, W], "counts": to_rle(m).encode()} for m in kept]

        gt = []
        for a in per_image.get(e["id"], []):
            rles = mu.frPyObjects(a["segmentation"], H, W)
            gt.append({"size": [H, W], "counts": to_rle(mu.decode(mu.merge(rles))).encode()})

        n_gt_total += len(gt)
        n_pred_total += len(pred)

        if pred and gt:
            iou = mu.iou(pred, gt, [0] * len(gt))  # shape (n_pred, n_gt)
            overlap_matrix = iou > OVERLAP_THRESH

            # one-to-many: a GT column matched by >1 prediction row
            matches_per_gt = overlap_matrix.sum(axis=0)
            n_frag = int((matches_per_gt > 1).sum())
            n_fragmented += n_frag
            if n_frag:
                frag_examples.append((e["file_name"], n_frag, int(matches_per_gt.max())))

            # many-to-one: a prediction row matched by >1 GT column
            matches_per_pred = overlap_matrix.sum(axis=1)
            n_merge = int((matches_per_pred > 1).sum())
            n_merged_preds += n_merge
            n_gt_involved_in_merge += int(overlap_matrix[matches_per_pred > 1].sum())
            if n_merge:
                merge_examples.append((e["file_name"], n_merge, int(matches_per_pred.max())))

            best_per_gt = iou.max(axis=0)
        else:
            best_per_gt = np.zeros(len(gt))

        m_tp = int((best_per_gt > 0.5).sum())
        tp += m_tp
        n_fp = len(pred) - m_tp
        n_fn = len(gt) - m_tp
        pq_num += float(best_per_gt[best_per_gt > 0.5].sum())
        pq_den += m_tp + 0.5 * n_fp + 0.5 * n_fn

        if i % 20 == 0:
            print(f"  {i}/{len(val_entries)}", flush=True)

    print(f"\n=== {len(val_entries)} val images, {n_gt_total} GT filaments, "
          f"{n_pred_total} predictions ===")
    print(f"PQ = {pq_num/pq_den:.4f}, TP = {tp}")
    print(f"\nOne-to-many (fragmentation): {n_fragmented}/{n_gt_total} GT filaments "
          f"({100*n_fragmented/n_gt_total:.1f}%) matched by >1 prediction at IoU>{OVERLAP_THRESH}")
    print(f"Many-to-one (over-merging): {n_merged_preds} predictions covering "
          f"{n_gt_involved_in_merge} GT filaments they shouldn't be merging "
          f"({100*n_merged_preds/n_pred_total:.1f}% of predictions)")

    if frag_examples:
        print(f"\nworst fragmentation examples (file, n_gt_fragmented, max_frags_per_instance):")
        for f in sorted(frag_examples, key=lambda x: -x[2])[:5]:
            print(f"  {f}")
    if merge_examples:
        print(f"\nworst over-merge examples (file, n_merged_preds, max_gt_per_pred):")
        for f in sorted(merge_examples, key=lambda x: -x[2])[:5]:
            print(f"  {f}")


if __name__ == "__main__":
    main()
