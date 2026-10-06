"""Out-of-fold numbers for the paper, all rows on the same five random 2-fold
calibration splits of the validation images (seed s -> the same fold
assignment for every row, so rows are paired per image).

    python paper_oof.py ROW          # compute one row -> paper_oof/ROW.npz
    python paper_oof.py --summarize  # table: OOF PQ mean (sd), SQ, RQ, paired-bootstrap deltas vs base4,
                                     # organizers' overlap statistics, human-vs-model on multi-annotator images

Per image and seed we store the official PQ counts (sum IoU of hits, TP, FP, FN; Self-Evaluation notebook
definition). For seed 1 we also store the organizers' descriptive statistics: IoU of every overlapping
(gt, pred) pair, and per-GT / per-prediction numbers of overlapping partners (one-to-many / many-to-one).
"""
import sys
from pathlib import Path

import numpy as np
import pycocotools.mask as mu

from eval_pooled_gt import official_counts
from sweep_ensemble_4way import H, W, paint_panoptic_rle
from sweep_mask_fusion import calibrated, fuse_image

OUT = Path("paper_oof")
SEEDS = [1, 2, 3, 4, 5]
DET = ["Av", "B1280v", "Cv", "L1280v"]
VOTE12 = DET + ["A", "B1280", "C", "L1280", "AF", "BF", "CF", "LF"]
NOFUSE, CFG = (2.0, 2.0, 1.0), (0.5, 0.3, 0.3)
ROWS = {  # name: (calibrated sources = voters, leaders, fusion config)
    "solo_A": (["Av"], ["Av"], NOFUSE),
    "solo_B": (["B1280v"], ["B1280v"], NOFUSE),
    "solo_C": (["Cv"], ["Cv"], NOFUSE),
    "solo_L": (["L1280v"], ["L1280v"], NOFUSE),
    "solo_S": (["S"], ["S"], NOFUSE),
    "base4": (DET, DET, NOFUSE),
    "plainS": (DET + ["S"], DET + ["S"], NOFUSE),
    "fuse4": (DET, DET, CFG),
    "fuse12": (VOTE12, DET, CFG),
    "lightS": (DET + ["S"], DET + ["S"], CFG),
    "fullS": (VOTE12 + ["S"], DET + ["S"], CFG),
}


def overlap_stats(pred, gt):
    """IoUs of overlapping pairs, per-GT and per-pred overlap degrees (organizers' m2n plot)."""
    if not pred or not gt:
        return np.zeros(0), np.zeros(len(gt), int), np.zeros(len(pred), int)
    p = [{"size": [H, W], "counts": r.encode()} for r in pred]
    g = [{"size": [H, W], "counts": r.encode()} for r in gt]
    iou = mu.iou(p, g, [0] * len(g))  # pred x gt
    hit = iou > 0
    return iou[hit], hit.sum(0), hit.sum(1)


def compute(row):
    import pickle
    sources, leaders, cfg = ROWS[row]
    cache = pickle.load(open("paper_val.pkl", "rb"))  # big_val.pkl slimmed to the 13 sources used here
    counts = np.zeros((len(SEEDS), len(cache), 4))
    ious, gdeg, pdeg = [], [], []
    for k, seed in enumerate(SEEDS):
        for i, (pooled, (_, gt)) in enumerate(zip(calibrated(cache, sources, True, seed), cache)):
            kept = paint_panoptic_rle(fuse_image(pooled, [cfg], leaders)[cfg])
            counts[k, i] = official_counts(kept, gt)
            if k == 0:
                a, b, c = overlap_stats(kept, gt)
                ious.append(a), gdeg.append(b), pdeg.append(c)
        print(f"{row} seed {seed} done", flush=True)
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / f"{row}.npz", counts=counts, ious=np.concatenate(ious),
             gdeg=np.concatenate(gdeg), pdeg=np.concatenate(pdeg))


NAMES = {"solo_A": "Mask R-CNN (2048\\,px)", "solo_B": "YOLO11m-seg (1280\\,px)", "solo_C": "RT-DETR-l (1280\\,px)",
         "solo_L": "YOLO11l-seg (1280\\,px)", "solo_S": "Semantic U-Net instances",
         "base4": "4 detectors, calibrated NMS", "plainS": "+ semantic, plain NMS", "fuse4": "+ cluster vote (4 voters)",
         "fuse12": "+ cluster vote (12 voters)", "lightS": "+ semantic leader (5 voters)",
         "fullS": "+ semantic leader (13 voters)"}
PUBLIC = {"base4": ".40", "fuse4": ".40", "fuse12": ".40", "fullS": ".40"}  # leaderboard, 2 decimals


def pq_of(c):  # c: (..., 4) summed over images
    return c[..., 0] / (c[..., 1] + 0.5 * c[..., 2] + 0.5 * c[..., 3])


def summarize(latex=False):
    latex_rows = []
    from collections import Counter
    from dataset import train_val_split
    data = {r: np.load(OUT / f"{r}.npz") for r in ROWS if (OUT / f"{r}.npz").exists()}
    base = data["base4"]["counts"]
    rng = np.random.default_rng(0)
    n = base.shape[1]
    boot = rng.integers(0, n, (2000, n))
    print(f"{'row':8s} {'PQ mean (sd)':>15s} {'SQ':>6s} {'RQ':>6s} {'TP':>6s}  delta vs base4: mean [min,max] "
          f"(#seeds with 95% CI > 0)")
    for r, d in data.items():
        c = d["counts"]
        tot = c.sum(1)  # per seed
        pq = pq_of(tot)
        sq = (tot[:, 0] / tot[:, 1]).mean()
        rq = (tot[:, 1] / (tot[:, 1] + 0.5 * tot[:, 2] + 0.5 * tot[:, 3])).mean()
        line = f"{r:8s} {pq.mean():.4f} ({pq.std(ddof=1):.4f}) {sq:.4f} {rq:.4f} {tot[:, 1].mean():6.1f}"
        if r != "base4":
            deltas, sig = [], 0
            for k in range(len(SEEDS)):
                diff = pq_of(c[k][boot].sum(1)) - pq_of(base[k][boot].sum(1))
                lo = np.percentile(diff, 2.5)
                deltas.append(pq[k] - pq_of(base[k].sum(0)))
                sig += lo > 0
            line += f"  {np.mean(deltas):+.4f} [{min(deltas):+.4f},{max(deltas):+.4f}] ({sig}/5)"
        print(line)
        if latex:
            f = lambda v: f"{v:.3f}".lstrip("0")
            cells = [NAMES[r], f"{f(pq.mean())} ({f(pq.std(ddof=1))})", f(sq), f(rq)]
            if r.startswith("solo"):
                cells.append("")
            elif r == "base4":
                cells.append("--")
            else:
                sgn = lambda v: ("$+$" if v >= 0 else "$-$") + f(abs(v))
                cells.append(f"[{sgn(min(deltas))}, {sgn(max(deltas))}] ({sig}/5)")
            cells.append(PUBLIC.get(r, ""))
            latex_rows.append(" & ".join(cells) + r" \\")
    if latex:
        print("\n".join(latex_rows))
    print("\nOrganizers' descriptive statistics (seed 1):")
    for r in ["base4", "fullS"]:
        d = data[r]
        ious = d["ious"]
        dice = 2 * ious / (1 + ious)
        g, p = Counter(d["gdeg"].tolist()), Counter(d["pdeg"].tolist())
        print(f"{r}: overlapping pairs {len(ious)}, mean IoU {ious.mean():.3f}, mean Dice {dice.mean():.3f}, "
              f"matched (IoU>0.5) pairs median IoU {np.median(ious[ious > 0.5]):.3f} "
              f"| GT degree 0/1/2+: {g[0]}/{g[1]}/{sum(v for k, v in g.items() if k >= 2)} "
              f"| pred degree 0/1/2+: {p[0]}/{p[1]}/{sum(v for k, v in p.items() if k >= 2)}")
    # human vs model on the same multi-annotator validation images
    _, val_entries, _ = train_val_split(val_frac=0.1, seed=0)
    files = [e["file_name"] for e in val_entries]
    multi = np.array([files.count(f) > 1 for f in files])
    for r in ["base4", "fullS"]:
        tot = data[r]["counts"][:, multi].sum(1)
        print(f"{r} on the {len(set(np.array(files)[multi]))} multi-annotator val images "
              f"({multi.sum()} annotation sets): OOF PQ {pq_of(tot).mean():.4f}")


if __name__ == "__main__":
    if sys.argv[1] == "--summarize":
        summarize(latex="--latex" in sys.argv)
    else:
        compute(sys.argv[1])
