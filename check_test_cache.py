"""Consistency check: the isotonic Experiment A pipeline replayed from the test
candidate cache must reproduce the rows predict_ensemble_expA.py wrote live
for the same images (identical masks, identical order)."""
import pickle
import sys

import pandas as pd

from sweep_ensemble_4way import CACHE_PATH, fit_calibrators, dedup_nms_rle, paint_panoptic_rle
from predict_ensemble_expA import SOURCES, ACCEPT, DEDUP_IOU

test_cache = sys.argv[1] if len(sys.argv) > 1 else "ensemble4_test_candidates.pkl.partial"
val = pickle.load(open(CACHE_PATH, "rb"))
cals = fit_calibrators(val, SOURCES, list(range(len(val))))
test = pickle.load(open(test_cache, "rb"))
sub = pd.read_csv("submission_expA.csv.partial", dtype=str)
done = set(open("submission_expA.csv.done").read().split())
stems = sub["filament_id"].str.rsplit("_", n=1).str[0]

n = ok = 0
for per_source, stem in test:
    if stem not in done:
        continue
    pooled = []
    for key in SOURCES:
        refined = per_source[key]
        if refined:
            cs = cals[key].predict([s for s, _, _ in refined])
            pooled += [(float(c), r) for c, (_, r, _) in zip(cs, refined) if c >= ACCEPT]
    kept = paint_panoptic_rle(dedup_nms_rle(pooled, DEDUP_IOU))
    expected = sub.loc[stems == stem, "segmentation_rle"].tolist()
    n += 1
    ok += kept == expected
    print(f"{stem}: cache={len(kept)} live={len(expected)} identical={kept == expected}")
print(f"{ok}/{n} images identical")
