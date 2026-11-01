#!/usr/bin/env bash
# Finish the all-data models after the laptop crashes: Mask R-CNN last epoch (~20 min), then the 512 px refiner (~2 h).
set -u
cd "/c/projects/kaggle sun thiing"
export PYTHONUTF8=1
echo "[$(date +%H:%M)] Mask R-CNN resume"
[ -f checkpoints/maskrcnn_hires2048_all_epoch5.pt ] || \
  python train_maskrcnn_hires.py --min-size 2048 --max-size 2048 --epochs 6 --all-data --resume checkpoints/maskrcnn_hires2048_all_epoch4.pt > train_maskrcnn_hires2048_all_resume.log 2>&1
echo "[$(date +%H:%M)] refiner 512"
[ -f checkpoints/refiner_v10_resnet34_c512_full_best.pt ] || \
  python train_refiner.py --spine --encoder resnet34 --lr 3e-4 --crop-size 512 --full-data --epochs 15 > train_refiner_v10c512_full.log 2>&1
echo "[$(date +%H:%M)] ALL DONE R2"
