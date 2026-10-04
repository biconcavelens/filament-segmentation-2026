"""Rebuild the deployed 4-source candidate caches (hires Mask R-CNN A, YOLO11m
B1280, RT-DETR C, YOLO11l L1280) with the pretrained-encoder refiner v10,
using the repo's own sweep_ensemble_4way.py. Outputs in /kaggle/working:
  v10_val_ABC.pkl / v10_test_ABC.pkl   (sources A, B1280, C)
  v10_val_L.pkl   / v10_test_L.pkl     (YOLO11l under key B1280 -> rename to L1280 when merging)
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "segmentation-models-pytorch"], check=True)
SRC = next(p.parent for p in Path("/kaggle/input").rglob("sweep_ensemble_4way.py"))
WORK = Path("/kaggle/working/src")
WORK.mkdir(parents=True, exist_ok=True)
for f in SRC.iterdir():
    if f.suffix == ".py":
        shutil.copy(f, WORK / f.name)
(WORK / "data").mkdir(exist_ok=True)
comp = next(Path("/kaggle/input").rglob("MAGFiLO_1.0_Kaggle_2026"))
link = WORK / "data" / "MAGFiLO_1.0_Kaggle_2026"
if not link.exists():
    link.symlink_to(comp)
os.chdir(WORK)

# per-run knobs (v1: refiner_v10_resnet34_best.pt/"v10", v2: refiner_v10_resnet34_c512_best.pt/"c512")
# v3 (RT-DETR-x, C only, "cx") collapsed in training and was never run
# v4 ("hf"): EXTRA=["--hflip"], horizontal-flip TTA sources
REFINER, TAG = "refiner_v10_resnet34_best.pt", "a2"
RTDETR = "rtdetr_cls_best.pt"
MASKRCNN = "maskrcnn_hires2048_s2_epoch5.pt"  # v5: second-seed hires Mask R-CNN, source A only
ONLY_C, ONLY_A = False, True
EXTRA = []
ck = {n: str(SRC / n) for n in ["maskrcnn_hires2048_epoch5.pt", "yolo11m_cls_best.pt", "yolo11l_cls_1280_best.pt",
                                 "rtdetr_cls_best.pt"]}
ck["refiner"] = str(next(Path("/kaggle/input").rglob(REFINER)))
ck["rtdetr_cls_best.pt"] = str(next(Path("/kaggle/input").rglob(RTDETR)))
ck["maskrcnn_hires2048_epoch5.pt"] = str(next(Path("/kaggle/input").rglob(MASKRCNN)))
OUT = Path("/kaggle/working")


def run(args):
    cmd = [sys.executable, "-u", "sweep_ensemble_4way.py", "--refiner", ck["refiner"],
           "--maskrcnn", ck["maskrcnn_hires2048_epoch5.pt"]] + EXTRA + args
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


# RT-DETR path is a module constant; patch it in the copy
s = (WORK / "sweep_ensemble_4way.py").read_text()
s = s.replace('RTDETR_CKPT = "kaggle_kernel_rtdetr_cls/output/rtdetr_cls_best.pt"', f'RTDETR_CKPT = "{ck["rtdetr_cls_best.pt"]}"')
(WORK / "sweep_ensemble_4way.py").write_text(s)

m, l = ck["yolo11m_cls_best.pt"], ck["yolo11l_cls_1280_best.pt"]
if ONLY_A:
    run(["--sources", "A", "--yolo", m, "--cache", str(OUT / f"{TAG}_val_A.pkl")])
    run(["--build-test-cache", "--sources", "A", "--yolo", m, "--test-cache", str(OUT / f"{TAG}_test_A.pkl")])
    print("ALL DONE", flush=True)
    sys.exit(0)
if ONLY_C:
    run(["--sources", "C", "--cache", str(OUT / f"{TAG}_val_C.pkl")])
    run(["--build-test-cache", "--sources", "C", "--test-cache", str(OUT / f"{TAG}_test_C.pkl")])
    print("ALL DONE", flush=True)
    sys.exit(0)
run(["--sources", "A", "B1280", "C", "--yolo", m, "--cache", str(OUT / f"{TAG}_val_ABC.pkl")])
run(["--sources", "B1280", "--yolo", l, "--cache", str(OUT / f"{TAG}_val_L.pkl")])
run(["--build-test-cache", "--sources", "A", "B1280", "C", "--yolo", m, "--test-cache", str(OUT / f"{TAG}_test_ABC.pkl")])
run(["--build-test-cache", "--sources", "B1280", "--yolo", l, "--test-cache", str(OUT / f"{TAG}_test_L.pkl")])
print("ALL DONE", flush=True)
