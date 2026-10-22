"""Merge the all-data test candidate caches into the 13-source layout of big_test.pkl (keys as in paper_val.pkl):
    v10 refiner (256 px):  A->Av  B1280->B1280v  C->Cv  L(B1280)->L1280v
    512 px refiner:        A->A   B1280->B1280   C->C   L(B1280)->L1280
    mirrored (hf):         A->AF  B1280->BF      C->CF  L(B1280)->LF
    semantic:              S (from `sweep_semseg.py --export ... --splits test` on the all-data model)

    python build_final_test_cache.py DIR_WITH_final_test_*.pkl SEMSEG_TEST_PKL OUT.pkl
"""
import pickle
import sys
from pathlib import Path

RENAME = {"v10": {"A": "Av", "B1280": "B1280v", "C": "Cv", "L": "L1280v"},
          "c512": {"A": "A", "B1280": "B1280", "C": "C", "L": "L1280"},
          "hf": {"A": "AF", "B1280": "BF", "C": "CF", "L": "LF"}}


def main():
    d, sem_path, out_path = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    sem = pickle.load(open(sem_path, "rb"))
    merged = [({}, stem) for _, stem in sem]
    for tag, names in RENAME.items():
        for part, src_keys in [("ABC", ["A", "B1280", "C"]), ("L", ["B1280"])]:
            cache = pickle.load(open(d / f"final_test_{tag}_{part}.pkl", "rb"))
            assert [s for _, s in cache] == [s for _, s in sem], f"image order mismatch in {tag}_{part}"
            for (ps, _), (acc, _) in zip(cache, merged):
                for k in src_keys:
                    acc[names["L" if part == "L" else k]] = ps[k]
    for (ps, _), (acc, _) in zip(sem, merged):
        acc["S"] = ps["S"]
    pickle.dump(merged, open(out_path, "wb"))
    n = {k: sum(len(a[k]) for a, _ in merged) for k in merged[0][0]}
    print(f"wrote {out_path}: {len(merged)} images; candidates per source: {n}")


if __name__ == "__main__":
    main()
