#!/usr/bin/env bash
# Final models trained on train + validation images (no held-out data left), laptop GPU, one job at a time.
# Detectors YOLO11m/l and RT-DETR go to Kaggle (kaggle_kernel_*, ALL_DATA=1); the calibration stays the one fitted
# on the held-out validation predictions of the models trained without validation.
set -u
cd "/c/projects/kaggle sun thiing"
export PYTHONUTF8=1
log() { echo "[$(date +%H:%M)] $*"; }

log "refiner v10 (ResNet34, 256 px crops), all data"
[ -f checkpoints/refiner_v10_resnet34_full_best.pt ] || \
  python train_refiner.py --spine --encoder resnet34 --lr 3e-4 --full-data --epochs 15 > train_refiner_v10_full.log 2>&1

log "Mask R-CNN 2048, all data"
[ -f checkpoints/maskrcnn_hires2048_all_epoch5.pt ] || \
  python train_maskrcnn_hires.py --min-size 2048 --max-size 2048 --epochs 6 --all-data > train_maskrcnn_hires2048_all.log 2>&1

log "semantic ConvNeXt-Tiny U-Net, all data (3 h)"
[ -f semseg_all/semseg.pt ] || \
  SEMSEG_OUT=semseg_all SEMSEG_ALL=1 SEMSEG_HOURS=3 SEMSEG_SEED=2 python train_semseg_local.py > semseg_all_train.log 2>&1

log "ALL DONE"
