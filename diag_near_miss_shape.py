"""For Experiment A's near-miss pairs (kept prediction whose best GT IoU is
0.3-0.5): is the prediction truncated (high precision, low recall -- covers
only part of the GT) or bloated (low precision, high recall)? Also: which
source produced it, and does the GT extend past the refiner's crop window
(square_bounds of the coarse proposal), which would cap recall regardless of
refiner quality."""
import pickle
from collections import Counter

import numpy as np
import pycocotools.mask as mu

from crop_dataset import square_bounds
from sweep_ensemble_4way import CACHE_PATH, H, W, fit_calibrators, dedup_nms_rle, paint_panoptic_rle

SOURCES, ACCEPT, DEDUP = ["A", "B1280"], 0.45, 0.05  # the real-0.39 2-way pipeline


def enc(r):
    return {"size": [H, W], "counts": r.encode()}


cache = pickle.load(open(CACHE_PATH, "rb"))
cals = fit_calibrators(cache, SOURCES, list(range(len(cache))))
rows = []
for per_source, gt in cache:
    pooled, src_of = [], {}
    for key in SOURCES:
        if per_source[key]:
            cs = cals[key].predict([s for s, _, _ in per_source[key]])
            for c, (_, r, _) in zip(cs, per_source[key]):
                if c >= ACCEPT:
                    pooled.append((float(c), r))
                    src_of.setdefault(r, key)
    kept_pre = dedup_nms_rle(pooled, DEDUP)  # pre-paint masks, so source lookup works
    kept = paint_panoptic_rle(kept_pre)
    if not kept or not gt:
        continue
    P, G = [enc(r) for r in kept], [enc(r) for r in gt]
    iou = mu.iou(P, G, [0] * len(G))
    a_p, a_g = mu.area(P).astype(float), mu.area(G).astype(float)
    # painting only removes pixels, so map each painted mask back to its pre-paint candidate by overlap
    pre = [enc(r) for _, r in kept_pre]
    back = mu.iou(P, pre, [1] * len(pre)).argmax(axis=1)  # max of intersection/area(painted)
    for i in range(len(P)):
        g = int(iou[i].argmax())
        if not 0.3 <= iou[i, g] < 0.5:
            continue
        inter = iou[i, g] * (a_p[i] + a_g[g]) / (1 + iou[i, g])
        pb = mu.toBbox([P[i]])[0]
        gb = mu.toBbox([G[g]])[0]
        # the refiner's crop window around this prediction: how much of the GT could it even see,
        # and what IoU would a perfect refiner (pred = GT inside the window) get?
        gm = mu.decode(G[g]).astype(bool)
        x0, y0, x1, y1 = square_bounds(mu.decode(P[i]).astype(bool))
        in_win = gm[y0:y1, x0:x1].sum() / gm.sum()
        rows.append(dict(in_window=in_win, ub_iou=in_win,  # pred = GT∩window -> IoU = |GT∩win| / |GT|
                         prec=inter / a_p[i], rec=inter / a_g[g], src=src_of[kept_pre[back[i]][1]],
                         area_ratio=a_p[i] / a_g[g],
                         gt_outside_pred_box=float(gb[0] < pb[0] - 5 or gb[1] < pb[1] - 5 or
                                                   gb[0] + gb[2] > pb[0] + pb[2] + 5 or
                                                   gb[1] + gb[3] > pb[1] + pb[3] + 5)))

prec = np.array([r["prec"] for r in rows])
rec = np.array([r["rec"] for r in rows])
print(f"near-miss pairs: {len(rows)}")
print(f"precision (inter/pred): median={np.median(prec):.2f}  recall (inter/gt): median={np.median(rec):.2f}")
print(f"truncated (prec>0.7, rec<0.6): {int(((prec > 0.7) & (rec < 0.6)).sum())}")
print(f"bloated   (rec>0.7, prec<0.6): {int(((rec > 0.7) & (prec < 0.6)).sum())}")
print(f"both low  (prec<0.7, rec<0.7): {int(((prec < 0.7) & (rec < 0.7)).sum())}")
print(f"area ratio pred/gt: median={np.median([r['area_ratio'] for r in rows]):.2f}")
print(f"GT extends >5px past the prediction's bbox: {int(sum(r['gt_outside_pred_box'] for r in rows))}")
print("source:", Counter(r["src"] for r in rows))
win = np.array([r["in_window"] for r in rows])
print(f"GT fraction inside the refiner's crop window: median={np.median(win):.2f}  "
      f">=0.9: {int((win >= 0.9).sum())}  >=0.7: {int((win >= 0.7).sum())}  <0.5: {int((win < 0.5).sum())}")
print(f"near misses a perfect in-window refiner would turn into TPs (IoU>0.5): {int((win > 0.5).sum())}/{len(rows)}")
