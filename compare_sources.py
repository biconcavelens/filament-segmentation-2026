"""A/B two versions of one candidate source (e.g. semantic models trained with/without neighbour
frames) on the same five out-of-fold calibration splits as paper_oof.py, alone and inside the
full system (detections from the four detectors + the source, masks voted by 13 sources).
Paired bootstrap over validation entries, per split.

    python compare_sources.py ab_val.pkl Sbase Snb
"""
import pickle
import sys

import numpy as np

from eval_pooled_gt import official_counts
from paper_oof import CFG, DET, NOFUSE, SEEDS, VOTE12, pq_of
from sweep_ensemble_4way import paint_panoptic_rle
from sweep_mask_fusion import calibrated, fuse_image


def counts(cache, sources, leaders, cfg):
    out = np.zeros((len(SEEDS), len(cache), 4))
    for k, seed in enumerate(SEEDS):
        for i, (pooled, (_, gt)) in enumerate(zip(calibrated(cache, sources, True, seed), cache)):
            out[k, i] = official_counts(paint_panoptic_rle(fuse_image(pooled, [cfg], leaders)[cfg]), gt)
    return out


def main():
    path, a, b = sys.argv[1:4]
    cache = pickle.load(open(path, "rb"))
    boot = np.random.default_rng(0).integers(0, len(cache), (2000, len(cache)))
    for name, setup in [("alone", lambda s: ([s], [s], NOFUSE)),
                        ("full system", lambda s: (VOTE12 + [s], DET + [s], CFG))]:
        ca, cb = counts(cache, *setup(a)), counts(cache, *setup(b))
        line = []
        for k in range(len(SEEDS)):
            d = pq_of(cb[k][boot].sum(1)) - pq_of(ca[k][boot].sum(1))
            line.append(f"{pq_of(cb[k].sum(0)) - pq_of(ca[k].sum(0)):+.4f} [{np.percentile(d, 2.5):+.4f},"
                        f"{np.percentile(d, 97.5):+.4f}]")
        print(f"{name}: {a} {pq_of(ca.sum(1)).mean():.4f}  {b} {pq_of(cb.sum(1)).mean():.4f}  "
              f"TP {ca.sum(1)[:, 1].mean():.0f} -> {cb.sum(1)[:, 1].mean():.0f}", flush=True)
        print(f"  {b} - {a} per split [95% CI]: " + "  ".join(line), flush=True)


if __name__ == "__main__":
    main()
