#!/usr/bin/env bash
# Third local stage: finish the all-data Mask R-CNN (the first run crashed with a cuDNN host-allocation error in the
# last epoch; epoch-4 checkpoint resumes it). Runs after the 512 px refiner so only one heavy job is on the GPU.
set -u
cd "/c/projects/kaggle sun thiing"
export PYTHONUTF8=1
until grep -q "ALL DONE 2" all_data_local2.log; do sleep 120; done
echo "[$(date +%H:%M)] Mask R-CNN 2048 all data, resume after epoch 4"
[ -f checkpoints/maskrcnn_hires2048_all_epoch5.pt ] || \
  python train_maskrcnn_hires.py --min-size 2048 --max-size 2048 --epochs 6 --all-data \
    --resume checkpoints/maskrcnn_hires2048_all_epoch4.pt > train_maskrcnn_hires2048_all_resume.log 2>&1
echo "[$(date +%H:%M)] ALL DONE 3"
