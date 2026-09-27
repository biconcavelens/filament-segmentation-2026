#!/usr/bin/env bash
# GONG SSL pipeline, run after download_gong.py finishes:
# SimSiam backbone -> Mask R-CNN@2048 fine-tune -> val candidates -> ensemble sweeps.
set -e
python -u pretrain_ssl.py --epochs 40 > pretrain_ssl.log 2>&1
python -u train_maskrcnn_hires.py --min-size 2048 --max-size 2048 --epochs 6 \
    --backbone checkpoints/ssl_simsiam_r50.pt > train_maskrcnn_ssl.log 2>&1
python sweep_ensemble_4way.py --sources A --maskrcnn checkpoints/maskrcnn_hires2048_ssl_epoch5.pt \
    --cache ens_ssl_A.pkl > ens_ssl_build.log 2>&1
python merge_candidate_caches.py ens_all_val.pkl ens_ssl_A.pkl ens_ssl_val.pkl A
for S in "A B1280 C" "A B1280 L1280 C"; do
    echo "== $S (SSL Mask R-CNN) =="
    python sweep_ensemble_4way.py --from-cache --cache ens_ssl_val.pkl --sources $S | grep -A2 'top 5'
done
