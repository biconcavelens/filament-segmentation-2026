"""Sweep the refiner's sigmoid threshold. A community reference (RF-DETR-Seg
notebook citing a Mask R-CNN+U-Net-refiner solution at ~0.69 LB) uses 0.85,
not the naive 0.5 -- mask-head/refiner blur on thin structures means a
stricter cutoff may recover a lot of boundary precision (near-miss errors).
"""
import numpy as np
import torch

import predict_refined as pr
from dataset import train_val_split
from sweep_thresh import pq_for

device = torch.device("cuda")
detector, refiner = pr.load_models("checkpoints/maskrcnn_epoch3.pt", "checkpoints/refiner_best.pt", device)
_, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)

for rt in [0.5, 0.65, 0.75, 0.85, 0.9, 0.95]:
    pr.REFINER_THRESHOLD = rt
    pq = pq_for(detector, refiner, device, val_entries, per_image)
    print(f"refiner_thresh={rt}: full-val(116) PQ={pq:.4f}", flush=True)
