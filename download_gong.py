"""Download unlabeled GONG H-alpha images for self-supervised pretraining.

Source: NSO's documented direct-access path for GONG H-alpha data
(https://nso.edu/data/nisp-data/h-alpha/ -> "Direct FTP access"),
https://gong2.nso.edu/ftp/HA/hag/ -- full-resolution 2048x2048 wavelet-
processed JPGs, the same processing as the competition images. The host
confirmed additional GONG H-alpha observations with no MAGFiLO annotations
may be used for self-supervised pretraining.

Polite by design: one request at a time with a pause, only the chosen .jpg
files and the listings needed to find them (listing pages also carry a
site-wide honeypot link; only timestamp-pattern .jpg names are parsed).
Days containing any competition test or val image are excluded so neither
split leaks in, even unlabeled. Only the six sites the competition uses.
Resumable: the plan is seeded, existing files are skipped.

    python download_gong.py --n 6000
"""
import argparse
import csv
import io
import math
import random
import re
import time
from pathlib import Path

import requests
from PIL import Image

from dataset import train_val_split

BASE = "https://gong2.nso.edu/ftp/HA/hag/"
SITES = set("BCLMTU")
TEST_DIR = Path("data/MAGFiLO_1.0_Kaggle_2026/test/test_images")
FRAME_RE = re.compile(r'href="(\d{14})([A-Z])h\.jpg"')
UA = "filament-segmentation-research/1.0 (non-commercial; Kaggle Solar Filament Segmentation Challenge 2026)"


class Polite:
    def __init__(self, pause):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA
        self.pause = pause
        self.fails = 0

    def get(self, url):
        for attempt in range(4):
            time.sleep(self.pause * (2 ** attempt))
            try:
                r = self.s.get(url, timeout=60)
                if r.status_code == 200:
                    self.fails = 0
                    return r
                if r.status_code in (403, 429):  # being told to back off: stop, don't push
                    raise SystemExit(f"server returned {r.status_code} for {url}; stopping")
            except requests.RequestException as e:
                print(f"  retry {attempt + 1} {url}: {e}", flush=True)
        self.fails += 1
        if self.fails >= 10:
            raise SystemExit("10 consecutive failures; stopping")
        return None


def excluded_days():
    _, val_entries, _ = train_val_split(val_frac=0.1, seed=0)
    days = {e["file_name"][:8] for e in val_entries}
    days |= {p.name[:8] for p in TEST_DIR.iterdir()}
    return days


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=6000)
    p.add_argument("--per-site", type=int, default=2,
                   help="frames per site per day, from different parts of that site's day; a day listing "
                        "takes ~20s to fetch (vs ~2.4s per image), so each listing should yield several")
    p.add_argument("--pause", type=float, default=1.0)
    p.add_argument("--out", default="data/gong_unlabeled")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.csv"
    have = {f.stem for f in out.glob("*.jpg")}
    skip_days = excluded_days()
    http = Polite(args.pause)
    rng = random.Random(args.seed)

    months = sorted(set(re.findall(r'href="(\d{6})/"', http.get(BASE).text)))
    per_month = math.ceil(args.n / len(months))
    first, last = months[0], months[-1]
    rng.shuffle(months)  # if the target is hit early, the cut falls randomly, not on the newest years
    days_per_month = math.ceil(per_month / (args.per_site * 4))  # ~4 of the 6 sites observe on a typical day
    print(f"{len(months)} months ({first}..{last}), ~{days_per_month} days/month, "
          f"{args.per_site} frames/site/day, {len(skip_days)} test/val days excluded, {len(have)} already on disk",
          flush=True)

    got = len(have)
    new_file = not manifest.exists()
    with open(manifest, "a", newline="") as mf:
        w = csv.writer(mf)
        if new_file:
            w.writerow(["stem", "site", "url"])
        for month in months:
            if got >= args.n:
                break
            r = http.get(f"{BASE}{month}/")
            if r is None:
                continue
            days = [d for d in sorted(set(re.findall(r'href="(\d{8})/"', r.text))) if d not in skip_days]
            for day in rng.sample(days, min(days_per_month, len(days))):
                r = http.get(f"{BASE}{month}/{day}/")
                if r is None:
                    continue
                by_site = {}
                for stamp, site in set(FRAME_RE.findall(r.text)):
                    if site in SITES:
                        by_site.setdefault(site, []).append(stamp)
                picks = []
                for site in sorted(by_site):
                    frames = sorted(by_site[site])
                    k = min(args.per_site, len(frames))
                    chunk = len(frames) / k  # one frame from each part of the site's observing day
                    picks += [(frames[int(i * chunk + rng.random() * chunk)], site) for i in range(k)]
                for stamp, site in picks:
                    stem = f"{stamp}{site}h"
                    if stem in have:
                        continue
                    url = f"{BASE}{month}/{day}/{stem}.jpg"
                    img = http.get(url)
                    if img is None:
                        continue
                    try:
                        im = Image.open(io.BytesIO(img.content))
                        if im.size != (2048, 2048):
                            continue
                    except Exception:
                        continue
                    (out / f"{stem}.jpg").write_bytes(img.content)
                    w.writerow([stem, site, url])
                    mf.flush()
                    have.add(stem)
                    got += 1
                    if got % 100 == 0:
                        print(f"  {got}/{args.n} images ({month})", flush=True)
                    if got >= args.n:
                        break
    print(f"done: {got} images in {out}", flush=True)


if __name__ == "__main__":
    main()
