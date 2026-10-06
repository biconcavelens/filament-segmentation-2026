#!/usr/bin/env bash
# Neighbour-frame A/B for the semantic model, fully chained (one heavy job at a time: RAM is tight).
# Baseline (semseg_ab_base) is started separately; this waits for it and for download_neighbors.py.
set -u
cd "/c/projects/kaggle sun thiing"
export PYTHONUTF8=1
log() { echo "[$(date +%H:%M)] $*"; }
trained() { grep -E "^epoch .*elapsed=(3\.[0-9]{2}|[4-9]\.[0-9]{2})h" "$1" >/dev/null 2>&1; }

log "waiting for neighbour download and baseline training"
until grep -q "^done:" download_neighbors.log && trained semseg_ab_base.log; do sleep 60; done
log "neighbour frames: $(($(grep -c . data/gong_neighbors/manifest.csv) - 1))"

log "training neighbour model"
SEMSEG_OUT=semseg_ab_nb SEMSEG_HOURS=3 SEMSEG_SEED=2 SEMSEG_NEIGHBORS=data/gong_neighbors/manifest.csv \
  python train_semseg_local.py > semseg_ab_nb.log 2>&1
trained semseg_ab_nb.log || { log "neighbour training did not finish; see semseg_ab_nb.log"; exit 1; }

for m in base nb; do
  log "CPU inference on val: $m"
  SEMSEG_OUT=semseg_ab_$m SEMSEG_PRED_TEST=0 python train_semseg_local.py predict > semseg_ab_${m}_predict.log 2>&1
  log "standalone grid: $m"
  python sweep_semseg.py --probs semseg_ab_$m/probs --t-fg 0.5 --t-sp 0.3 --close 3 --min-area 100 \
    --score-floor 0.0 > semseg_ab_${m}_grid.log 2>&1
  python sweep_semseg.py --probs semseg_ab_$m/probs --export 0.5 0.3 3 100 --key S$m --splits val \
    >> semseg_ab_${m}_grid.log 2>&1
done
python merge_candidate_caches.py paper_val.pkl semseg_Sbase_val.pkl ab_tmp.pkl Sbase > /dev/null
python merge_candidate_caches.py ab_tmp.pkl semseg_Snb_val.pkl ab_val.pkl Snb
log "comparing (5 splits, paired bootstrap)"
python compare_sources.py ab_val.pkl Sbase Snb 2>&1 | grep -v -i warn > neighbor_ab_result.log
cat neighbor_ab_result.log
log "ALL DONE"
