#!/usr/bin/env bash
# Kaggle runs two GPU sessions at a time: start the YOLO11l all-data kernel as soon as YOLO11m or RT-DETR is done.
cd "/c/projects/kaggle sun thiing"; export PYTHONUTF8=1
while true; do
  for s in filament-all-yolo11m filament-all-rtdetr; do
    st=$(kaggle kernels status trishanthmellimi/$s 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z]*' | head -1)
    case "$st" in *COMPLETE*|*ERROR*|*CANCEL*) echo "[$(date +%H:%M)] $s $st -> pushing yolo11l"
      kaggle kernels push -p kaggle_kernel_all_yolo11l 2>&1 | tail -1; exit 0;; esac
  done
  sleep 240
done
