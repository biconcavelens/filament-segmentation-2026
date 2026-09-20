"""Full-test-set prediction: solo YOLO26m-seg, default settings (no cls=
hack applied yet). Real-leaderboard data point for the untuned architecture,
established locally at PQ=0.4217 (thresh=0.35) -- weaker than our existing
cls-fixed solo detectors (YOLO11: 0.4252, RT-DETR: 0.4296), so expected to
land below the current best (0.39) rather than beat it. Submitted anyway
per the same "document everything, positive or negative" discipline used
all session (see RESULTS.md).
"""
import numpy as np
import pandas as pd
import torch
from PIL import Image
from ultralytics import YOLO

from dataset import H, W
from train_refiner import RefinerUNet
from crop_dataset import square_bounds, CROP_SIZE
from predict_trained import paint_panoptic, to_rle
from predict_refined import refine_with_tta

YOLO26_CKPT = "kaggle_kernel_yolo26/output/yolo26m_best.pt"
REFINER_CKPT = "checkpoints/refiner_v5_best.pt"
CONF_THRESH = 0.35  # confirmed local peak, sweep_yolo26_solo.py
MIN_AREA = 20

D = __import__("pathlib").Path("data/MAGFiLO_1.0_Kaggle_2026")
TEST_DIR = D / "test" / "test_images"


@torch.no_grad()
def predict_one(yolo, refiner, device, gray, img_path):
    out = yolo.predict(source=str(img_path), imgsz=1280, conf=CONF_THRESH, verbose=False)[0]
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


def main():
    device = torch.device("cuda")
    yolo = YOLO(YOLO26_CKPT)
    rstate = torch.load(REFINER_CKPT, map_location=device)
    refiner = RefinerUNet(in_channels=rstate.get("in_channels", 1),
                           out_channels=rstate.get("out_channels", 1)).to(device)
    refiner.load_state_dict(rstate["model"])
    refiner.eval()

    files = sorted(TEST_DIR.iterdir())
    rows = []
    for i, path in enumerate(files, 1):
        gray = np.array(Image.open(path).convert("L"))
        kept = predict_one(yolo, refiner, device, gray, path)
        stem = path.stem
        rows.extend({"filament_id": f"{stem}_{k}", "segmentation_rle": to_rle(m)}
                    for k, m in enumerate(kept, 1))
        if i % 20 == 0:
            print(f"{i}/{len(files)} test images done", flush=True)

    out_df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    out_df.to_csv("submission_yolo26_solo.csv", index=False)
    print(f"\nwrote submission_yolo26_solo.csv: {len(out_df)} rows, {len(files)} images, "
          f"avg {len(out_df)/len(files):.1f} filaments/image", flush=True)


if __name__ == "__main__":
    main()
