"""Human-level reference: on training images annotated by 2+ people, score one
annotator's filaments against another's with the official PQ logic (every
pair with IoU > 0.5 is a TP). This bounds how well any model can agree with a
single annotator, and splits the disagreement into recognition (which
filaments are marked) vs segmentation (where their boundaries are).

    python diag_human_pq.py
"""
import itertools
from collections import defaultdict

import numpy as np
import pycocotools.mask as mu

from dataset import train_val_split, H, W
from eval_pooled_gt import official_counts, pq


def main():
    train, val, per_image = train_val_split(val_frac=0.1, seed=0)
    by_file = defaultdict(list)
    for e in train + val:
        by_file[e["file_name"]].append(e)
    multi = {f: es for f, es in by_file.items() if len(es) > 1}
    tot = np.zeros(4)
    n_pairs = 0
    for f, es in multi.items():
        rles = []
        for e in es:
            rs = []
            for a in per_image.get(e["id"], []):
                m = mu.decode(mu.merge(mu.frPyObjects(a["segmentation"], H, W)))
                rs.append(mu.encode(np.asfortranarray(m))["counts"].decode())
            rles.append(rs)
        for i, j in itertools.permutations(range(len(es)), 2):  # both directions, like pred-vs-gt
            tot += official_counts(rles[i], rles[j])
            n_pairs += 1
    s, tp, fp, fn = tot
    sq = s / tp if tp else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn)
    print(f"{len(multi)} multi-annotator images, {n_pairs} ordered annotator pairs")
    print(f"human PQ={pq(tot):.4f}  SQ={sq:.4f}  RQ={rq:.4f}  (TP {int(tp)} FP {int(fp)} FN {int(fn)})")


if __name__ == "__main__":
    main()
