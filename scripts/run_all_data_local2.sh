#!/usr/bin/env bash
# Second local stage: the 512 px refiner on all data (runs after run_all_data_local.sh; ~2 h on the laptop GPU).
set -u
cd "/c/projects/kaggle sun thiing"
export PYTHONUTF8=1
until grep -q "ALL DONE" all_data_local.log; do sleep 120; done
echo "[$(date +%H:%M)] refiner v10 (ResNet34, 512 px crops), all data"
[ -f checkpoints/refiner_v10_resnet34_c512_full_best.pt ] || \
  python train_refiner.py --spine --encoder resnet34 --lr 3e-4 --crop-size 512 --full-data --epochs 15 > train_refiner_v10c512_full.log 2>&1
echo "[$(date +%H:%M)] ALL DONE 2"
