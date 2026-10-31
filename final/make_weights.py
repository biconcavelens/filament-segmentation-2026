"""Pack the deployed (0.40 submission) weights for the repo: float32 tensors stored as float16 (every file < 100 MB,
so plain git works) and the isotonic calibrators as plain arrays. predict.py casts back to float32 on load.

    python final/make_weights.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "final" / "weights"
OUT.mkdir(exist_ok=True)

SOURCES = ["Av", "B1280v", "Cv", "L1280v", "A", "B1280", "C", "L1280", "AF", "BF", "CF", "LF", "S"]
FP32 = {"maskrcnn.pt": "kaggle_ds_v10/maskrcnn_hires2048_epoch5.pt",
        "refiner256.pt": "kaggle_ds_v10/refiner_v10_resnet34_best.pt",
        "refiner512.pt": "kaggle_ds_refiners/refiner_v10_resnet34_c512_best.pt",
        "semseg.pt": "kaggle_out_semseg/semseg.pt"}
ALREADY_HALF = {"yolo11m.pt": "kaggle_ds_v10/yolo11m_cls_best.pt", "rtdetr.pt": "kaggle_ds_v10/rtdetr_cls_best.pt",
                "yolo11l.pt": "kaggle_ds_v10/yolo11l_cls_1280_best.pt"}


def half(sd):
    return {k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in sd.items()}


for name, src in FP32.items():
    st = torch.load(ROOT / src, map_location="cpu")
    st["model"] = half(st["model"])
    torch.save(st, OUT / name)
for name, src in ALREADY_HALF.items():
    (OUT / name).write_bytes((ROOT / src).read_bytes())

from sweep_ensemble_4way import fit_calibrators  # noqa: E402  (calibrators fitted on the held-out validation predictions)
val = pickle.load(open(ROOT / "paper_val.pkl", "rb"))
cals = fit_calibrators(val, SOURCES, list(range(len(val))))
json.dump({k: {"x": c.X_thresholds_.tolist(), "y": c.y_thresholds_.tolist()} for k, c in cals.items()},
          open(OUT / "calibrators.json", "w"))
for k, c in cals.items():  # np.interp on the stored thresholds must equal the sklearn calibrator
    s = np.linspace(0, 1, 501)
    assert np.allclose(c.predict(s), np.interp(s, c.X_thresholds_, c.y_thresholds_), atol=1e-9), k
for f in sorted(OUT.iterdir()):
    print(f"{f.stat().st_size / 2**20:7.1f} MB  {f.name}")
