#!/usr/bin/env bash
# Restart after the laptop went down on Oct 22 ~19:40: semantic test inference (resumable), then 512 px refiner, then Mask R-CNN resume.
set -u
cd "/c/projects/kaggle sun thiing"
export PYTHONUTF8=1
echo "[$(date +%H:%M)] semantic test inference"
SEMSEG_OUT=semseg_all SEMSEG_ALL=1 SEMSEG_PRED_TEST=1 python train_semseg_local.py predict > semseg_all_predict2.log 2>&1
echo "[$(date +%H:%M)] refiner 512"
[ -f checkpoints/refiner_v10_resnet34_c512_full_best.pt ] || \
  python train_refiner.py --spine --encoder resnet34 --lr 3e-4 --crop-size 512 --full-data --epochs 15 > train_refiner_v10c512_full.log 2>&1
echo "[$(date +%H:%M)] Mask R-CNN resume"
[ -f checkpoints/maskrcnn_hires2048_all_epoch5.pt ] || \
  python train_maskrcnn_hires.py --min-size 2048 --max-size 2048 --epochs 6 --all-data --resume checkpoints/maskrcnn_hires2048_all_epoch4.pt > train_maskrcnn_hires2048_all_resume.log 2>&1
echo "[$(date +%H:%M)] ALL DONE R"
