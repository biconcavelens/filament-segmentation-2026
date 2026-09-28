"""After run_ssl_pipeline.sh: build test candidates for the SSL Mask R-CNN,
pick the better of the 3-way / 4-way ensembles by val PQ, check fusion on val,
and write + validate the fused submission (submission_ssl_fusion.csv).
"""
import re
import subprocess
import sys

PY = sys.executable
MASKRCNN = "checkpoints/maskrcnn_hires2048_ssl_epoch5.pt"
CONFIGS = {"A B1280 C": None, "A B1280 L1280 C": None}


def run(cmd):
    print(">>", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def best_from_sweep(sources):
    out = subprocess.run(f"{PY} sweep_ensemble_4way.py --from-cache --cache ens_ssl_val.pkl --sources {sources}",
                         shell=True, check=True, capture_output=True, text=True).stdout
    top = out.split("=== top 5 ===")[1].strip().splitlines()[0]
    m = re.match(r"PQ=([\d.]+)\s+dedup_iou=([\d.]+)\s+accept=([\d.]+)", top)
    return float(m[1]), float(m[2]), float(m[3])


def main():
    run(f"{PY} sweep_ensemble_4way.py --build-test-cache --sources A --maskrcnn {MASKRCNN} "
        f"--test-cache ens_ssl_test_A.pkl > ens_ssl_test_build.log 2>&1")
    run(f"{PY} merge_candidate_caches.py ens_all_test.pkl ens_ssl_test_A.pkl ens_ssl_test.pkl A")
    for s in CONFIGS:
        CONFIGS[s] = best_from_sweep(s)
        print(f"{s}: val PQ={CONFIGS[s][0]:.4f} dedup={CONFIGS[s][1]} accept={CONFIGS[s][2]}", flush=True)
    sources = max(CONFIGS, key=lambda s: CONFIGS[s][0])
    pq, dedup, accept = CONFIGS[sources]
    print(f"chosen: {sources} (val PQ {pq:.4f})", flush=True)
    run(f"{PY} sweep_semantic_fusion.py --probs unet_val_probs.npy --cache ens_ssl_val.pkl --sources {sources} "
        f"--accept {accept} --dedup {dedup} --ts 0.7 --gs 100 > ssl_fusion_val.log 2>&1")
    print(open("ssl_fusion_val.log").read().split("images\n")[-1], flush=True)
    run(f"{PY} predict_fusion.py --unet checkpoints/unet_best.pt --cache ens_ssl_val.pkl "
        f"--test-cache ens_ssl_test.pkl --sources {sources} --accept {accept} --dedup {dedup} "
        f"--t-sem 0.7 --grow 100 --out submission_ssl_fusion.csv")
    run(f"{PY} validate_submission.py submission_ssl_fusion.csv")
    import pandas as pd
    dups = pd.read_csv("submission_ssl_fusion.csv").filament_id.duplicated().sum()
    print(f"dup ids: {dups}", flush=True)
    sys.exit(1 if dups else 0)


if __name__ == "__main__":
    main()
