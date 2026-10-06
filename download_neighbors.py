"""Download GONG H-alpha frames taken a few minutes before/after each TRAINING image, from the
same site. Filaments move < 1 px in that time (measured shift 0.1-0.3 px at +-2..5 min), so a
neighbour frame can be trained on with the labels of its training image: the same filaments
under different seeing. Host ruling: sequences may be used in training; inference uses only
the supplied test image.

GONG frames of a site sit at a fixed second within each minute, so neighbour names are
predicted instead of fetching day listings (~20 s each). One request at a time with a pause;
only predicted .jpg names are requested (listing pages carry a honeypot link and are not used).

Never downloaded: val or test images themselves, and any frame within 30 min of a val or test
image from the same site (so neither split leaks into training, even approximately).

    python download_neighbors.py --offsets -4 -2 2 4
"""
import argparse
import csv
import datetime as dt
import io
import time
from pathlib import Path

import requests
from PIL import Image

from dataset import train_val_split
from download_gong import BASE, TEST_DIR, UA

T = lambda stem: dt.datetime.strptime(stem[:14], "%Y%m%d%H%M%S")  # noqa: E731


def protected():
    """(site -> times) of val and test images: no frame within 30 min of these is fetched."""
    _, val, _ = train_val_split(val_frac=0.1, seed=0)
    stems = {e["file_name"][:16] for e in val} | {p.name[:16] for p in TEST_DIR.iterdir()}
    out = {}
    for s in stems:
        out.setdefault(s[14], []).append(T(s))
    return out


def too_close(stem, prot, minutes=30):
    return any(abs((T(stem) - t).total_seconds()) < minutes * 60 for t in prot.get(stem[14], []))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--offsets", type=int, nargs="+", default=[-4, -2, 2, 4], help="minutes")
    p.add_argument("--pause", type=float, default=1.0)
    p.add_argument("--out", default="data/gong_neighbors")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.csv"
    done = set()
    if manifest.exists():
        done = {(r["source"], int(r["offset"])) for r in csv.DictReader(open(manifest))}
    train, _, _ = train_val_split(val_frac=0.1, seed=0)
    sources = sorted({e["file_name"] for e in train})
    prot = protected()
    s = requests.Session()
    s.headers["User-Agent"] = UA
    got = miss = fails = 0
    new = not manifest.exists()
    with open(manifest, "a", newline="") as mf:
        w = csv.writer(mf)
        if new:
            w.writerow(["stem", "source", "offset"])
        for n, src in enumerate(sources):
            t0, site = T(src), src[14]
            for off in args.offsets:
                if (src, off) in done:
                    continue
                for jitter in (0, 1 if off > 0 else -1):  # target minute, then one further out
                    stem = (t0 + dt.timedelta(minutes=off + jitter)).strftime("%Y%m%d%H%M%S") + site + "h"
                    if too_close(stem, prot):
                        break
                    time.sleep(args.pause)
                    try:
                        r = s.get(f"{BASE}{stem[:6]}/{stem[:8]}/{stem}.jpg", timeout=60)
                    except requests.RequestException:
                        fails += 1
                        continue
                    if r.status_code in (403, 429):
                        raise SystemExit(f"server returned {r.status_code}; stopping")
                    if r.status_code != 200:
                        continue
                    try:
                        if Image.open(io.BytesIO(r.content)).size != (2048, 2048):
                            continue
                    except Exception:
                        continue
                    (out / f"{stem}.jpg").write_bytes(r.content)
                    w.writerow([stem, src, off])
                    mf.flush()
                    got += 1
                    break
                else:
                    miss += 1
                if fails >= 20:
                    raise SystemExit("20 network failures; stopping")
            if (n + 1) % 50 == 0:
                print(f"{n + 1}/{len(sources)} sources: {got} frames, {miss} offsets missing", flush=True)
    print(f"done: {got} frames, {miss} offsets with no frame", flush=True)


if __name__ == "__main__":
    prot = {"B": [dt.datetime(2014, 6, 9, 20, 0, 0)]}
    assert too_close("20140609201000Bh", prot) and not too_close("20140609203100Bh", prot)
    assert not too_close("20140609201000Ch", prot)
    main()
