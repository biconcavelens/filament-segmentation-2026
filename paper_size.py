"""Detection quality by filament size for the paper (one out-of-fold calibration split, seed 1, as paper_thin.py).

For every annotated filament: area and best IoU with any kept prediction (matched if > 0.5).
For every kept prediction: area and whether it is matched. Bins are quartiles of the annotated areas.

    python paper_size.py            # -> paper_oof/size.npz + printed summary
"""
import pickle

import numpy as np
import pycocotools.mask as mu

from paper_oof import ROWS
from sweep_ensemble_4way import H, W, paint_panoptic_rle
from sweep_mask_fusion import calibrated, fuse_image

SEED = 1
ORDER = ["base4", "fuse12", "fullS"]


def rles(rs):
    return [{"size": [H, W], "counts": r.encode()} for r in rs]


def main():
    cache = pickle.load(open("paper_val.pkl", "rb"))
    gt_area = np.concatenate([mu.area(rles(gt)) if gt else np.zeros(0) for _, gt in cache]).astype(float)
    out = {"gt_area": gt_area}
    for name in ORDER:
        sources, leaders, cfg = ROWS[name]
        best, p_area, p_hit = [], [], []
        for pooled, (_, gt) in zip(calibrated(cache, sources, True, SEED), cache):
            kept = paint_panoptic_rle(fuse_image(pooled, [cfg], leaders)[cfg])
            iou = mu.iou(rles(kept), rles(gt), [0] * len(gt)) if kept and gt else np.zeros((len(kept), len(gt)))
            best += list(iou.max(0)) if len(kept) else [0.0] * len(gt)
            p_area += list(mu.area(rles(kept))) if kept else []
            p_hit += list(iou.max(1) > 0.5) if len(gt) else [False] * len(kept)
        out[f"{name}_best"], out[f"{name}_parea"], out[f"{name}_phit"] = np.array(best), np.array(p_area, float), np.array(p_hit)
        print(name, "done", flush=True)
    np.savez("paper_oof/size.npz", **out)
    summarize(out)


def summarize(d):
    edges = np.quantile(d["gt_area"], [0, .25, .5, .75, 1])
    edges[-1] = np.inf
    print("GT area quartile edges (px):", edges[1:-1].round())
    for name in ORDER:
        b, pa, ph = d[f"{name}_best"], d[f"{name}_parea"], d[f"{name}_phit"]
        cells = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            g = (d["gt_area"] >= lo) & (d["gt_area"] < hi)
            p = (pa >= lo) & (pa < hi)
            m = b[g] > 0.5
            cells.append(f"recall {m.mean():.3f} SQ {b[g][m].mean():.3f} prec {ph[p].mean():.3f} (n_gt {g.sum()}, n_pred {p.sum()})")
        print(f"{name:7s} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
