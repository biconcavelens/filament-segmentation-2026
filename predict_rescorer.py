"""Submission from the learned candidate rescorer (sweep_rescorer.py): the GBM
is trained on all 116 cached val images, then scores the cached test
candidates (build them with `sweep_ensemble_4way.py --build-test-cache
--sources ...`), followed by the same true-NMS dedup + panoptic paint.

    python predict_rescorer.py --sources A B1280 B2048 --accept 0.45 --out submission_rescorer.csv
    python predict_rescorer.py --isotonic --accept 0.5 --out submission_expA.csv   # Experiment A
"""
import argparse
import pickle

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier

from dataset import train_val_split, IMG_DIR
from sweep_ensemble_4way import (CACHE_PATH, TEST_CACHE_PATH, TEST_DIR, ALL_SOURCES,
                                 dedup_nms_rle, paint_panoptic_rle, fit_calibrators)
from sweep_rescorer import image_features, GBM_PARAMS


def gray(path):
    return np.array(Image.open(path).convert("L"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="+", default=["A", "B1280", "B2048"], choices=ALL_SOURCES)
    p.add_argument("--accept", type=float, required=True)
    p.add_argument("--dedup", type=float, default=0.05)
    p.add_argument("--contrast", action="store_true")
    p.add_argument("--isotonic", action="store_true", help="per-source isotonic scores instead of the GBM")
    p.add_argument("--out", required=True)
    p.add_argument("--cache", default=CACHE_PATH, help="val candidate cache (fits the scorer)")
    p.add_argument("--test-cache", default=TEST_CACHE_PATH)
    args = p.parse_args()

    with open(args.cache, "rb") as f:
        val = pickle.load(f)
    with open(args.test_cache, "rb") as f:
        test = pickle.load(f)
    assert all(s in test[0][0] for s in args.sources), f"test cache lacks one of {args.sources}"

    if args.isotonic:
        cals = fit_calibrators(val, args.sources, list(range(len(val))))
    else:
        _, val_entries, _ = train_val_split(val_frac=0.1, seed=0)
        val_grays = ((gray(IMG_DIR / e["file_name"]) if args.contrast else None) for e in val_entries)
        feats = [f for f in (image_features(ps, args.sources, g) for (ps, _), g in zip(val, val_grays))
                 if f is not None]
        gbm = HistGradientBoostingClassifier(**GBM_PARAMS)
        gbm.fit(np.vstack([f[0] for f in feats]), np.concatenate([f[1] for f in feats]))

    rows = []
    for per_source, stem in test:
        g = gray(next(TEST_DIR.glob(stem + ".*"))) if args.contrast else None
        f = image_features(per_source, args.sources, g)
        if f is None:
            continue
        if args.isotonic:
            probs = np.array([cals[args.sources[s]].predict([v])[0] for s, v in zip(f[3], f[4])])
        else:
            probs = gbm.predict_proba(f[0])[:, 1]
        kept = paint_panoptic_rle(dedup_nms_rle(
            [(float(s), r) for s, r in zip(probs, f[2]) if s >= args.accept], args.dedup))
        rows.extend({"filament_id": f"{stem}_{k}", "segmentation_rle": r} for k, r in enumerate(kept, 1))

    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(rows)} rows, {len(test)} images, avg {len(rows) / len(test):.1f}/image")


if __name__ == "__main__":
    main()
