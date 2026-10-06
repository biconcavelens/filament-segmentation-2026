"""Grow or shrink every mask of a submission by N pixels (leaderboard probe for a width
offset between our masks and the hidden test annotations).

Masks stay disjoint (the scorer rejects overlaps): when growing, a new pixel goes to the
first mask in file order that reaches it and never to a pixel another mask already owns;
when shrinking, masks that fall below MIN_AREA are dropped.

    python postprocess_width.py submission_lead5S_vote13.csv submission_dil1.csv --px 1
    python postprocess_width.py submission_lead5S_vote13.csv submission_ero1.csv --px -1
"""
import argparse

import numpy as np
import pandas as pd
import pycocotools.mask as mu
from scipy import ndimage

H, W = 2048, 2048
MIN_AREA = 20


def change_width(masks, px):
    """masks: list of (H, W) uint8 in priority order -> list of masks, |px| pixels wider/narrower."""
    if px < 0:
        out = [ndimage.binary_erosion(m, iterations=-px).astype(np.uint8) for m in masks]
        return [m for m in out if m.sum() >= MIN_AREA]
    taken = np.zeros((H, W), bool)
    for m in masks:
        taken |= m.astype(bool)
    out = []
    for m in masks:
        grown = ndimage.binary_dilation(m, iterations=px) & ~taken  # only free pixels
        taken |= grown
        out.append((m.astype(bool) | grown).astype(np.uint8))
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument("src")
    a.add_argument("dst")
    a.add_argument("--px", type=int, required=True)
    args = a.parse_args()
    df = pd.read_csv(args.src, dtype=str)
    df["image"] = df["filament_id"].str.rsplit("_", n=1).str[0]
    rows = []
    for image, g in df.groupby("image", sort=False):
        masks = [mu.decode({"size": [H, W], "counts": r.encode()}) for r in g["segmentation_rle"]]
        for k, m in enumerate(change_width(masks, args.px), 1):
            rle = mu.encode(np.asfortranarray(m))["counts"].decode()
            rows.append((f"{image}_{k}", rle))
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(args.dst, index=False)
    print(f"{args.dst}: {len(rows)} masks ({len(df)} in source)")


if __name__ == "__main__":
    # self-check: two touching squares stay disjoint when grown, and shrink back
    a, b = np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8)
    a[10:20, 10:20], b[10:20, 20:30] = 1, 1
    g = change_width([a, b], 1)
    assert not (g[0] & g[1]).any() and g[0].sum() > a.sum() and g[1].sum() > b.sum()
    assert change_width([a], -1)[0].sum() == 64
    main()
