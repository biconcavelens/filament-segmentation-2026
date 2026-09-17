"""Two-detector candidate-union ensemble on the full test set (see
ensemble_diag.py for the local-val validation: PQ 0.390 -> 0.400).

Usage:
    python predict_ensemble2.py --out submission.csv
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from train import build_from_checkpoint
from train_refiner import RefinerUNet
from predict_trained import to_rle
from ensemble_diag import predict_ensemble

D = Path("data/MAGFiLO_1.0_Kaggle_2026")
TEST_DIR = D / "test" / "test_images"
CHUNK_SIZE = 20
COOLDOWN_SECONDS = 15


def load_models(device):
    stateA = torch.load("checkpoints/maskrcnn_full_epoch7.pt", map_location=device)
    detA = build_from_checkpoint(stateA, num_classes=2).to(device)
    detA.load_state_dict(stateA["model"]); detA.roi_heads.nms_thresh = 0.30; detA.eval()

    stateB = torch.load("checkpoints/maskrcnn_tile_epoch7.pt", map_location=device)
    detB = build_from_checkpoint(stateB, num_classes=2).to(device)
    detB.load_state_dict(stateB["model"]); detB.roi_heads.nms_thresh = 0.30; detB.eval()

    rstate = torch.load("checkpoints/refiner_v5_full_best.pt", map_location=device)
    refiner = RefinerUNet(in_channels=rstate.get("in_channels", 1),
                           out_channels=rstate.get("out_channels", 1)).to(device)
    refiner.load_state_dict(rstate["model"]); refiner.eval()
    return detA, detB, refiner


def predict(out_csv: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detA, detB, refiner = load_models(device)
    print(f"loaded detA=maskrcnn_full_epoch7 detB=maskrcnn_tile_epoch7 refiner=refiner_v5_full_best on {device}")

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
            gray = np.array(Image.open(path).convert("L"))
            rgb = np.array(Image.open(path).convert("RGB"))
            img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            kept = predict_ensemble(detA, detB, refiner, device, gray, img_t)
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


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="submission.csv")
    args = p.parse_args()
    predict(args.out)
