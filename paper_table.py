"""Numbers for the paper's results table: val PQ / SQ / RQ (calibration fit on
all val, in-sample) for single sources and each pipeline stage, from big_val.pkl.

    python paper_table.py
"""
import pickle

import numpy as np

from eval_pooled_gt import official_counts, pq
from sweep_ensemble_4way import paint_panoptic_rle
from sweep_mask_fusion import calibrated, fuse_image

DET = ["Av", "B1280v", "Cv", "L1280v"]
VOTE12 = DET + ["A", "B1280", "C", "L1280", "AF", "BF", "CF", "LF"]
NOFUSE, CFG = (2.0, 2.0, 1.0), (0.5, 0.3, 0.3)
ROWS = [  # name, sources (calibrated + voters), leaders, fusion config
    ("Mask R-CNN (2048)", ["Av"], ["Av"], NOFUSE),
    ("YOLO11m-seg (1280)", ["B1280v"], ["B1280v"], NOFUSE),
    ("RT-DETR-l (1280)", ["Cv"], ["Cv"], NOFUSE),
    ("YOLO11l-seg (1280)", ["L1280v"], ["L1280v"], NOFUSE),
    ("Semantic U-Net (S)", ["S"], ["S"], NOFUSE),
    ("4 detectors, NMS dedup", DET, DET, NOFUSE),
    ("+ cluster fusion, 4 voters", DET, DET, CFG),
    ("+ cluster fusion, 12 voters", VOTE12, DET, CFG),
    ("+ S as leader, 5 voters", DET + ["S"], DET + ["S"], CFG),
    ("+ S as leader, 13 voters", VOTE12 + ["S"], DET + ["S"], CFG),
]


def main():
    cache = pickle.load(open("big_val.pkl", "rb"))
    for name, sources, leaders, cfg in ROWS:
        tot = np.zeros(4)
        for pooled, (_, gt) in zip(calibrated(cache, sources, crossfit=False), cache):
            tot += official_counts(paint_panoptic_rle(fuse_image(pooled, [cfg], leaders)[cfg]), gt)
        s, tp, fp, fn = tot
        print(f"{name:32s} PQ={pq(tot):.4f} SQ={s / tp:.4f} RQ={tp / (tp + 0.5 * fp + 0.5 * fn):.4f} "
              f"TP={int(tp)} FP={int(fp)} FN={int(fn)}", flush=True)


if __name__ == "__main__":
    main()
