import numpy as np
import torch

import predict_refined as pr
from dataset import train_val_split
from sweep_thresh import pq_for

device = torch.device("cuda")
detector, refiner = pr.load_models("checkpoints/maskrcnn_epoch3.pt", "checkpoints/refiner_best.pt", device)
_, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)

for nms, thresh, label in [(0.5, 0.8, "current default"), (0.3, 0.8, "candidate winner")]:
    detector.roi_heads.nms_thresh = nms
    pr.DETECTOR_SCORE_THRESHOLD = thresh
    pq = pq_for(detector, refiner, device, val_entries, per_image)
    print(f"{label}: nms={nms} thresh={thresh}  full-val(116) PQ={pq:.4f}", flush=True)
