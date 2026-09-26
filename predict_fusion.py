"""Submission with semantic + instance fusion (sweep_semantic_fusion.py):
isotonic-calibrated pooled candidates from the cached test candidates ->
true-NMS dedup -> panoptic paint -> grow/add from the U-Net's probability map.

    python predict_fusion.py --unet checkpoints/unet_best.pt --t-sem 0.5 --grow 40 --out submission_fusion.csv
"""
import argparse
import pickle

import numpy as np
import pandas as pd
import torch
from PIL import Image

from predict_unet import load_model, predict_prob_map
from sweep_ensemble_4way import TEST_DIR, fit_calibrators, dedup_nms_rle, paint_panoptic_rle
from sweep_semantic_fusion import fuse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--unet", required=True)
    p.add_argument("--cache", default="ens_hires_e5.pkl")
    p.add_argument("--test-cache", default="ens_hires_e5_test_withC.pkl")
    p.add_argument("--sources", nargs="+", default=["A", "B1280", "C"])
    p.add_argument("--accept", type=float, default=0.5)
    p.add_argument("--dedup", type=float, default=0.05)
    p.add_argument("--t-sem", type=float, required=True)
    p.add_argument("--grow", type=int, required=True)
    p.add_argument("--new-min-area", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    val = pickle.load(open(args.cache, "rb"))
    test = pickle.load(open(args.test_cache, "rb"))
    cals = fit_calibrators(val, args.sources, list(range(len(val))))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = load_model(args.unet, device)

    rows = []
    for per_source, stem in test:
        pooled = []
        for key in args.sources:
            if per_source[key]:
                cs = cals[key].predict([s for s, _, _ in per_source[key]])
                pooled += [(float(c), r) for c, (_, r, _) in zip(cs, per_source[key]) if c >= args.accept]
        kept = paint_panoptic_rle(dedup_nms_rle(pooled, args.dedup))
        gray = np.array(Image.open(next(TEST_DIR.glob(stem + ".*"))).convert("L"))
        prob = predict_prob_map(unet, device, gray)
        fused = fuse(kept, prob, args.t_sem, args.grow, args.new_min_area)
        rows.extend({"filament_id": f"{stem}_{k}", "segmentation_rle": r} for k, r in enumerate(fused, 1))

    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(rows)} rows, {len(test)} images, avg {len(rows) / len(test):.1f}/image")


if __name__ == "__main__":
    main()
