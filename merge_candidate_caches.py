"""Swap one source's candidates into a val candidate cache, e.g. a retrained
Mask R-CNN's "A" into the existing cache so the other sources stay
byte-identical (and don't need re-running):

    python merge_candidate_caches.py base.pkl donor.pkl out.pkl A
"""
import pickle
import sys

base_path, donor_path, out_path, *sources = sys.argv[1:]
base = pickle.load(open(base_path, "rb"))
donor = pickle.load(open(donor_path, "rb"))
assert len(base) == len(donor) and all(b[1] == d[1] for b, d in zip(base, donor)), "val order/GT mismatch"
merged = [({**b[0], **{s: d[0][s] for s in sources}}, b[1]) for b, d in zip(base, donor)]
pickle.dump(merged, open(out_path, "wb"))
for s in merged[0][0]:
    c = [x for ps, _ in merged for x in ps[s]]
    print(f"{s}: {len(c)} candidates, {sum(l for *_, l in c)} TP-labelled")
