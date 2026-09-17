"""Local Panoptic Quality metric, matching the competition's definition:
PQ = sum(IoU over matches) / (|TP| + 0.5|FP| + 0.5|FN|), match = IoU > 0.5.

IoU>0.5 matching is automatically one-to-one (two predictions can't both
exceed 0.5 IoU with the same disjoint ground-truth segment), so no separate
assignment step is needed.
"""
import numpy as np
import pycocotools.mask as mu


def compute_pq(pred_rles: list[str], gt_rles: list[str], h: int = 2048, w: int = 2048) -> float:
    if not pred_rles and not gt_rles:
        return 1.0
    if not pred_rles or not gt_rles:
        return 0.0

    pred = [{"size": [h, w], "counts": r.encode("utf-8")} for r in pred_rles]
    gt = [{"size": [h, w], "counts": r.encode("utf-8")} for r in gt_rles]
    iou = mu.iou(pred, gt, [0] * len(gt))  # [n_pred, n_gt]

    matched_pred, matched_gt = set(), set()
    sum_iou = 0.0
    pairs = [(iou[i, j], i, j) for i in range(len(pred)) for j in range(len(gt))
             if iou[i, j] > 0.5]
    for v, i, j in sorted(pairs, reverse=True):
        if i in matched_pred or j in matched_gt:
            continue
        matched_pred.add(i)
        matched_gt.add(j)
        sum_iou += v

    tp = len(matched_pred)
    fp = len(pred) - tp
    fn = len(gt) - tp
    denom = tp + 0.5 * fp + 0.5 * fn
    return sum_iou / denom if denom > 0 else 1.0


def selftest():
    # perfect match
    m = np.zeros((10, 10), dtype=np.uint8)
    m[2:5, 2:5] = 1
    rle = mu.encode(np.asfortranarray(m))["counts"].decode("utf-8")
    assert compute_pq([rle], [rle], 10, 10) == 1.0

    # empty prediction against non-empty gt -> 0
    assert compute_pq([], [rle], 10, 10) == 0.0

    # both empty -> 1 (nothing to find, nothing predicted)
    assert compute_pq([], [], 10, 10) == 1.0

    # one clean TP + one spurious FP -> denom = 1 + 0.5
    m2 = np.zeros((10, 10), dtype=np.uint8)
    m2[7:9, 7:9] = 1
    rle2 = mu.encode(np.asfortranarray(m2))["counts"].decode("utf-8")
    pq = compute_pq([rle, rle2], [rle], 10, 10)
    assert abs(pq - (1.0 / 1.5)) < 1e-6, pq

    print("selftest OK")


if __name__ == "__main__":
    selftest()
