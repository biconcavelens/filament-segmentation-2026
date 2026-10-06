"""Figures for the SABiD paper.
  fig_pipeline.pdf     -- pipeline diagram
  fig_qualitative.pdf  -- one val image: GT vs 4-detector NMS baseline vs final fusion

    python paper/make_figures.py
"""
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mu
from matplotlib.patches import FancyBboxPatch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import train_val_split, IMG_DIR, H, W  # noqa: E402
from sweep_ensemble_4way import paint_panoptic_rle  # noqa: E402
from sweep_mask_fusion import calibrated, fuse_image  # noqa: E402

OUT = Path(__file__).resolve().parent
DET = ["Av", "B1280v", "Cv", "L1280v"]
VOTE = DET + ["A", "B1280", "C", "L1280", "AF", "BF", "CF", "LF", "S"]


def box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec="#333", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.2)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=0.8, color="#333"))


def pipeline():
    fig, ax = plt.subplots(figsize=(7.0, 2.35))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.3)
    ax.axis("off")
    box(ax, 0.05, 1.25, 1.1, 0.8, "GONG H$\\alpha$\nimage\n2048$\\times$2048", "#eef3fb")
    dets = ["Mask R-CNN (2048)", "YOLO11m-seg (1280)", "RT-DETR-l (1280)", "YOLO11l-seg (1280)"]
    for i, d in enumerate(dets):
        y = 2.55 - i * 0.55
        box(ax, 1.55, y, 1.6, 0.42, d, "#fdf2e3")
        arrow(ax, 1.15, 1.65, 1.55, y + 0.21)
        arrow(ax, 3.15, y + 0.21, 3.55, 1.95)
    box(ax, 1.55, 0.12, 1.6, 0.55, "Semantic U-Net\n(fg + spine) $\\to$ watershed", "#e8f5e9")
    arrow(ax, 1.15, 1.45, 1.55, 0.4)
    box(ax, 3.55, 1.45, 1.55, 1.0, "Crop refiner\n(ImageNet ResNet34\nU-Net, spine aux.,\n256/512 px, flip TTA)", "#fdf2e3")
    box(ax, 5.5, 1.45, 1.4, 1.0, "Per-source\nisotonic\ncalibration\nP(TP | score)", "#f3e5f5")
    arrow(ax, 5.1, 1.95, 5.5, 1.95)
    arrow(ax, 3.15, 0.4, 5.5, 1.6)
    box(ax, 7.3, 1.45, 1.25, 1.0, "Leaders:\ncross-source\nNMS (IoU 0.05)\naccept $\\geq$0.5", "#e3f2fd")
    arrow(ax, 6.9, 1.95, 7.3, 1.95)
    box(ax, 7.3, 0.12, 1.25, 1.0, "Voters:\nall candidates\nIoU>0.5 with a\nleader, score $\\geq$0.3", "#e3f2fd")
    arrow(ax, 6.9, 1.7, 7.3, 0.8)
    box(ax, 8.85, 0.85, 1.1, 1.2, "Weighted\npixel vote\n(share $\\geq$0.3)\n$\\to$ panoptic\npaint", "#fff8e1")
    arrow(ax, 8.55, 1.95, 8.85, 1.6)
    arrow(ax, 8.55, 0.6, 8.85, 1.1)
    fig.tight_layout(pad=0.1)
    fig.savefig(OUT / "fig_pipeline.pdf")
    fig.savefig(OUT / "fig_pipeline.png", dpi=200)


def contours(ax, rles, color, lw=0.9):
    for r in rles:
        m = mu.decode({"size": [H, W], "counts": r.encode()})
        ax.contour(m, levels=[0.5], colors=[color], linewidths=lw)


def qualitative(idx):
    cache = pickle.load(open(OUT.parent / "big_val.pkl", "rb"))
    _, val_entries, _ = train_val_split(val_frac=0.1, seed=0)
    pooled_base = calibrated(cache, DET, crossfit=False)[idx]
    pooled_all = calibrated(cache, VOTE, crossfit=False)[idx]
    base = paint_panoptic_rle(fuse_image(pooled_base, [(2.0, 2.0, 1.0)], DET)[(2.0, 2.0, 1.0)])
    final = paint_panoptic_rle(fuse_image(pooled_all, [(0.5, 0.3, 0.3)], DET + ["S"])[(0.5, 0.3, 0.3)])
    gt = cache[idx][1]
    img = np.array(Image.open(IMG_DIR / val_entries[idx]["file_name"]).convert("L"))
    ys, xs = np.where(sum(mu.decode({"size": [H, W], "counts": r.encode()}) for r in gt + final) > 0)
    pad = 60
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, H)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, W)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.65))
    for ax, (title, rles, col) in zip(axes, [("Annotation", gt, "#00e676"), ("4-detector NMS baseline", base, "#ff9100"),
                                            ("Final (fusion + semantic leader)", final, "#40c4ff")]):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        contours(ax, rles, col)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_title(f"{title} ({len(rles)})", fontsize=8)
        ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.01, top=0.9, wspace=0.03)
    fig.savefig(OUT / "fig_qualitative.pdf", dpi=300)
    fig.savefig(OUT / "fig_qualitative.png", dpi=200)
    print("qualitative entry", idx, val_entries[idx]["file_name"], "gt", len(gt), "base", len(base), "final", len(final))


def calibration():
    """Isotonic P(TP | raw score) per source (fit on all validation candidates)."""
    from sklearn.isotonic import IsotonicRegression
    cache = pickle.load(open(OUT.parent / "paper_val.pkl", "rb"))
    names = {"Av": "Mask R-CNN", "B1280v": "YOLO11m-seg", "Cv": "RT-DETR-l", "L1280v": "YOLO11l-seg",
             "S": "Semantic U-Net"}
    fig, ax = plt.subplots(figsize=(3.45, 2.1))
    xs = np.linspace(0, 1, 201)
    for (key, name), col in zip(names.items(), ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]):
        sc = np.array([s for ps, _ in cache for s, _, _ in ps[key]])
        lab = np.array([l for ps, _ in cache for _, _, l in ps[key]])
        iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(sc, lab)
        ax.plot(xs, iso.predict(xs), color=col, lw=1.3, label=name)
    ax.axhline(0.5, color="#888", lw=0.7, ls="--")
    ax.set_xlabel("raw detector score", fontsize=8)
    ax.set_ylabel("calibrated P(TP)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "fig_calibration.pdf")
    fig.savefig(OUT / "fig_calibration.png", dpi=200)


def overlap_figure():
    """Organizers' descriptive statistics (seed-1 out-of-fold run): IoU of every overlapping
    (prediction, annotation) pair, and per-annotation / per-prediction numbers of overlapping partners."""
    d = {r: np.load(OUT.parent / "paper_oof" / f"{r}.npz") for r in ["base4", "fullS"]}
    lab = {"base4": "4-detector NMS", "fullS": "final"}
    col = {"base4": "#ff9100", "fullS": "#1e88e5"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.45, 1.75), gridspec_kw={"width_ratios": [1.15, 1]})
    bins = np.linspace(0, 1, 21)
    for r in d:
        a.hist(d[r]["ious"], bins=bins, histtype="step", lw=1.2, color=col[r], label=lab[r])
    a.axvline(0.5, color="#888", lw=0.7, ls="--")
    a.set_xlabel("IoU of overlapping pairs", fontsize=7)
    a.set_ylabel("pairs", fontsize=7)
    a.tick_params(labelsize=6)
    a.legend(fontsize=5.5, frameon=False, loc="upper left")
    groups = ["GT 0", "GT 1", "GT 2+", "pred 0", "pred 1", "pred 2+"]
    x = np.arange(len(groups))
    for k, r in enumerate(d):
        g, p = d[r]["gdeg"], d[r]["pdeg"]
        vals = [(g == 0).sum(), (g == 1).sum(), (g >= 2).sum(), (p == 0).sum(), (p == 1).sum(), (p >= 2).sum()]
        b.bar(x + (k - 0.5) * 0.38, vals, width=0.38, color=col[r])
    b.set_xticks(x)
    b.set_xticklabels(groups, fontsize=5.5, rotation=40)
    b.set_ylabel("count", fontsize=7)
    b.tick_params(axis="y", labelsize=6)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "fig_overlap.pdf")
    fig.savefig(OUT / "fig_overlap.png", dpi=200)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "pipeline"):
        pipeline()
    if what in ("all", "qualitative"):
        qualitative(83)
    if what in ("all", "calibration"):
        calibration()
    if what in ("all", "overlap"):
        overlap_figure()
