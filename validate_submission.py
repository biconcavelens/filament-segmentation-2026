"""Pre-submission validator: catches the two failure modes reported on the
discussion board (SubmissionStatus.ERROR from overlapping masks; ill-formed
RLE) before burning a submission slot.

Usage:
    python validate_submission.py submission.csv
    python validate_submission.py --selftest
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pycocotools.mask as mu

H, W = 2048, 2048


def decode(rle_counts: str) -> np.ndarray:
    return mu.decode({"size": [H, W], "counts": rle_counts.encode("utf-8")})


def validate(csv_path: str) -> bool:
    df = pd.read_csv(csv_path, dtype=str)
    ok = True

    missing_cols = {"filament_id", "segmentation_rle"} - set(df.columns)
    if missing_cols:
        print(f"[FAIL] missing columns: {missing_cols}")
        return False

    dupes = df["filament_id"][df["filament_id"].duplicated()].tolist()
    if dupes:
        print(f"[FAIL] duplicate filament_id values: {dupes[:10]}"
              f"{' ...' if len(dupes) > 10 else ''}")
        ok = False

    df["image_id"] = df["filament_id"].str.rsplit("_", n=1).str[0]

    for image_id, group in df.groupby("image_id"):
        claimed = np.zeros((H, W), dtype=np.int16)
        for _, row in group.iterrows():
            try:
                mask = decode(row["segmentation_rle"])
            except Exception as e:
                print(f"[FAIL] {row['filament_id']}: RLE decode error: {e}")
                ok = False
                continue
            if mask.shape != (H, W):
                print(f"[FAIL] {row['filament_id']}: decoded shape "
                      f"{mask.shape} != ({H}, {W})")
                ok = False
                continue
            if mask.sum() == 0:
                print(f"[WARN] {row['filament_id']}: empty mask")
            claimed += mask.astype(np.int16)

        overlap_px = int((claimed > 1).sum())
        if overlap_px:
            print(f"[FAIL] image {image_id}: {overlap_px} overlapping "
                  f"pixels across {len(group)} filaments")
            ok = False

    n_images = df["image_id"].nunique()
    print(f"\nrows: {len(df)}  images: {n_images}  "
          f"avg filaments/image: {len(df) / n_images:.1f}")
    print("[PASS] no overlaps, all RLE valid" if ok else "[FAIL] see above")
    return ok


def _make_rle(mask: np.ndarray) -> str:
    return mu.encode(np.asfortranarray(mask))["counts"].decode("utf-8")


def selftest():
    a = np.zeros((H, W), dtype=np.uint8)
    a[0:10, 0:10] = 1
    b = np.zeros((H, W), dtype=np.uint8)
    b[5:15, 5:15] = 1  # overlaps a
    c = np.zeros((H, W), dtype=np.uint8)
    c[100:110, 100:110] = 1  # disjoint

    tmp = Path("scratch")
    tmp.mkdir(exist_ok=True)

    bad_csv = tmp / "_selftest_overlap.csv"
    pd.DataFrame({
        "filament_id": ["img1_1", "img1_2"],
        "segmentation_rle": [_make_rle(a), _make_rle(b)],
    }).to_csv(bad_csv, index=False)
    assert validate(str(bad_csv)) is False, "expected overlap to fail"

    good_csv = tmp / "_selftest_clean.csv"
    pd.DataFrame({
        "filament_id": ["img1_1", "img1_2"],
        "segmentation_rle": [_make_rle(a), _make_rle(c)],
    }).to_csv(good_csv, index=False)
    assert validate(str(good_csv)) is True, "expected disjoint to pass"

    dup_csv = tmp / "_selftest_dupe.csv"
    pd.DataFrame({
        "filament_id": ["img1_1", "img1_1"],
        "segmentation_rle": [_make_rle(a), _make_rle(c)],
    }).to_csv(dup_csv, index=False)
    assert validate(str(dup_csv)) is False, "expected duplicate id to fail"

    for f in (bad_csv, good_csv, dup_csv):
        f.unlink()
    print("\nselftest OK")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        selftest()
    elif len(sys.argv) == 2:
        sys.exit(0 if validate(sys.argv[1]) else 1)
    else:
        print(__doc__)
        sys.exit(1)
