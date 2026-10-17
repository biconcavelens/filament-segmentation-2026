"""Two cheap post-hoc ideas on the full system, 5 out-of-fold splits, paired bootstrap vs the current fullS:
  (1) source-specific vote weights (weight = calibrated score x multiplier of the voting source)
  (2) smoothing of the voted masks (fill holes / morphological closing)

    python exp_vote_weights_smooth.py
"""
import pickle

import numpy as np
import pycocotools.mask as mu
from scipy import ndimage

import sweep_mask_fusion as smf
from eval_pooled_gt import official_counts
from paper_oof import CFG, DET, SEEDS, VOTE12, pq_of
from sweep_ensemble_4way import H, W, paint_panoptic_rle
from sweep_mask_fusion import calibrated, to_rle

SOURCES, LEADERS = VOTE12 + ["S"], DET + ["S"]
WEIGHTS = {  # name -> multiplier per source (default 1)
    "S x2": {"S": 2.0}, "S x0.5": {"S": 0.5},
    "detectors x2": {k: 2.0 for k in DET}, "512px x0.5": {k: 0.5 for k in ["A", "B1280", "C", "L1280"]},
    "flips x0.5": {k: 0.5 for k in ["AF", "BF", "CF", "LF"]},
}
SMOOTH = {"fill holes": lambda m: ndimage.binary_fill_holes(m),
          "close r2": lambda m: ndimage.binary_closing(m, structure=np.ones((3, 3)), iterations=2),
          "fill + close r1": lambda m: ndimage.binary_closing(ndimage.binary_fill_holes(m), structure=np.ones((3, 3)))}


def fuse(pooled, mult):
    """fuse_image(pooled, [CFG], LEADERS) but with vote weights score*mult[source]."""
    if not pooled:
        return []
    rles = [{"size": [H, W], "counts": r.encode()} for _, r, _ in pooled]
    scores = np.array([s for s, _, _ in pooled])
    wts = scores * np.array([mult.get(src, 1.0) for _, _, src in pooled])
    iou = mu.iou(rles, rles, [0] * len(rles))
    leaders = []
    for i in range(len(pooled)):
        if pooled[i][2] in LEADERS and scores[i] >= smf.ACCEPT and all(iou[i, j] <= smf.DEDUP for j in leaders):
            leaders.append(i)
    fuse_iou, floor, vote = CFG
    members = {i: [i] for i in leaders}
    for j in range(len(pooled)):
        if j in members or scores[j] < floor or not leaders:
            continue
        k = int(np.argmax(iou[j, leaders]))
        if iou[j, leaders[k]] > fuse_iou:
            members[leaders[k]].append(j)
    boxes, out = mu.toBbox(rles), []
    for i in leaders:
        if len(members[i]) == 1:
            out.append((scores[i], pooled[i][1]))
            continue
        b = boxes[members[i]]
        x0, y0 = int(b[:, 0].min()), int(b[:, 1].min())
        x1, y1 = int(np.ceil((b[:, 0] + b[:, 2]).max())), int(np.ceil((b[:, 1] + b[:, 3]).max()))
        share = sum(wts[j] * mu.decode(rles[j])[y0:y1, x0:x1] for j in members[i]) / wts[members[i]].sum()
        m = np.zeros((H, W), np.uint8)
        m[y0:y1, x0:x1] = (share >= vote - 1e-9) & (share > 0)
        out.append((scores[i], to_rle(m)))
    return out


def smooth(kept, fn):
    out = []
    for r in kept:
        m = fn(mu.decode({"size": [H, W], "counts": r.encode()}).astype(bool)).astype(np.uint8)
        out.append(to_rle(m))
    return out


def main():
    cache = pickle.load(open("paper_val.pkl", "rb"))
    names = ["base"] + list(WEIGHTS) + list(SMOOTH)
    tot = {n: np.zeros((len(SEEDS), len(cache), 4)) for n in names}
    for k, seed in enumerate(SEEDS):
        for i, (pooled, (_, gt)) in enumerate(zip(calibrated(cache, SOURCES, True, seed), cache)):
            base_fused = fuse(pooled, {})
            tot["base"][k, i] = official_counts(paint_panoptic_rle(base_fused), gt)
            for n, mult in WEIGHTS.items():
                tot[n][k, i] = official_counts(paint_panoptic_rle(fuse(pooled, mult)), gt)
            for n, fn in SMOOTH.items():
                sm = [(s, r2) for (s, _), r2 in zip(base_fused, smooth([r for _, r in base_fused], fn))]
                tot[n][k, i] = official_counts(paint_panoptic_rle(sm), gt)
        print(f"seed {seed} done", flush=True)
    boot = np.random.default_rng(0).integers(0, len(cache), (2000, len(cache)))
    print(f"{'variant':16s} {'PQ':>7s}  diff vs base per split (CI excludes 0 marked *)")
    for n in names:
        c, line = tot[n], []
        for k in range(len(SEEDS)):
            d = pq_of(c[k][boot].sum(1)) - pq_of(tot["base"][k][boot].sum(1))
            lo, hi = np.percentile(d, [2.5, 97.5])
            line.append(f"{pq_of(c[k].sum(0)) - pq_of(tot['base'][k].sum(0)):+.4f}{'*' if lo > 0 or hi < 0 else ' '}")
        print(f"{n:16s} {pq_of(c.sum(1)).mean():.4f}  " + " ".join(line), flush=True)


if __name__ == "__main__":
    main()
