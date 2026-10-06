"""Fine-structure and near-miss analysis for the paper (one out-of-fold calibration split, seed 1).

Thin structures: the pixels of an annotated filament removed by a morphological opening with a
disk of radius r (r=2: parts narrower than ~5 px, which include barbs and faint tails; r=3 as a
sensitivity check). For each configuration we measure the fraction of thin and of core (opened)
annotated pixels covered by (a) any prediction and (b) the matched prediction of a detected
filament (IoU > 0.5), and record every annotated filament's best IoU, for the near-miss analysis.

    python paper_thin.py            # -> paper_oof/thin.npz + printed summary
"""
import pickle

import numpy as np
import pycocotools.mask as mu
from scipy import ndimage

from paper_oof import ROWS, DET, NOFUSE
from sweep_ensemble_4way import H, W, paint_panoptic_rle
from sweep_mask_fusion import calibrated, fuse_image

SEED = 1
CONFIGS = dict(ROWS)
CONFIGS["base4_512"] = (["A", "B1280", "C", "L1280"], ["A", "B1280", "C", "L1280"], NOFUSE)  # 512 px refiner
ORDER = ["base4", "base4_512", "fuse4", "fuse12", "fullS", "solo_S"]


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def gt_parts(gt_rles, radii=(2, 3)):
    """Per annotated filament: bbox and, per radius, (thin, core) boolean crops."""
    out = []
    for r in gt_rles:
        rle = {"size": [H, W], "counts": r.encode()}
        x, y, w, h = [int(v) for v in mu.toBbox(rle)]
        pad = 4
        y0, y1, x0, x1 = max(y - pad, 0), min(y + h + pad, H), max(x - pad, 0), min(x + w + pad, W)
        m = mu.decode(rle)[y0:y1, x0:x1].astype(bool)
        parts = {}
        for rad in radii:
            core = ndimage.binary_opening(m, structure=disk(rad))
            parts[rad] = (m & ~core, core)
        out.append(((y0, y1, x0, x1), m, parts))
    return out


def main():
    cache = pickle.load(open("paper_val.pkl", "rb"))
    gts = [gt_parts(gt) for _, gt in cache]
    res = {}
    for name in ORDER:
        sources, leaders, cfg = CONFIGS[name]
        pooled_all = calibrated(cache, sources, True, SEED)
        acc = {rad: np.zeros(8) for rad in (2, 3)}  # thin_any, thin_tot, core_any, core_tot, thin_m, thin_mtot, core_m, core_mtot
        best = []
        for pooled, (_, gt_rles), parts in zip(pooled_all, cache, gts):
            kept = paint_panoptic_rle(fuse_image(pooled, [cfg], leaders)[cfg])
            preds = [mu.decode({"size": [H, W], "counts": r.encode()}).astype(bool) for r in kept]
            union = np.any(preds, axis=0) if preds else np.zeros((H, W), bool)
            if kept and gt_rles:
                iou = mu.iou([{"size": [H, W], "counts": r.encode()} for r in kept],
                             [{"size": [H, W], "counts": r.encode()} for r in gt_rles], [0] * len(gt_rles))
            else:
                iou = np.zeros((len(kept), len(gt_rles)))
            for g, ((y0, y1, x0, x1), m, p) in enumerate(parts):
                b = float(iou[:, g].max()) if len(kept) else 0.0
                best.append(b)
                u = union[y0:y1, x0:x1]
                matched = preds[int(np.argmax(iou[:, g]))][y0:y1, x0:x1] if b > 0.5 else None
                for rad, (thin, core) in p.items():
                    a = acc[rad]
                    a[0] += (thin & u).sum(); a[1] += thin.sum(); a[2] += (core & u).sum(); a[3] += core.sum()
                    if matched is not None:
                        a[4] += (thin & matched).sum(); a[5] += thin.sum()
                        a[6] += (core & matched).sum(); a[7] += core.sum()
            del preds, union
        res[name] = (acc, np.array(best))
        a2, a3 = acc[2], acc[3]
        print(f"{name:10s} r=2: thin cov {a2[0] / a2[1]:.3f} core cov {a2[2] / a2[3]:.3f} | matched: thin {a2[4] / a2[5]:.3f}"
              f" core {a2[6] / a2[7]:.3f} | r=3 matched thin {a3[4] / a3[5]:.3f} | thin share of GT px "
              f"{a2[1] / (a2[1] + a2[3]):.3f} | TP {(np.array(best) > 0.5).sum()}", flush=True)
    np.savez("paper_oof/thin.npz", **{f"{n}_acc2": res[n][0][2] for n in res}, **{f"{n}_acc3": res[n][0][3] for n in res},
             **{f"{n}_best": res[n][1] for n in res})
    # near misses: annotated filaments with best IoU in [0.3, 0.5) under the 4-detector baseline
    b0, b1 = res["base4"][1], res["fullS"][1]
    near = (b0 >= 0.3) & (b0 < 0.5)
    print(f"near misses under base4: {near.sum()}; matched under final: {(b1[near] > 0.5).sum()}; "
          f"baseline matches lost by final: {((b0 > 0.5) & (b1 <= 0.5)).sum()}; "
          f"new matches from no/low overlap (<0.3): {((b0 < 0.3) & (b1 > 0.5)).sum()}; "
          f"mean best IoU of baseline matches: {b0[b0 > 0.5].mean():.3f} -> {b1[b0 > 0.5].mean():.3f}")
    for n in ["fuse4", "fuse12"]:
        b = res[n][1]
        print(f"  {n}: near misses converted {(b[near] > 0.5).sum()}, baseline matches lost {((b0 > 0.5) & (b <= 0.5)).sum()}")


if __name__ == "__main__":
    main()
