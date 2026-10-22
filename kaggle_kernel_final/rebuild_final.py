"""Final (all-data) test candidate caches: the models trained on train + validation images, run on the 180 test
images with the same recipe as the v10 / c512 / hf rebuilds (repo's sweep_ensemble_4way.py). Test only: the
calibration stays the one fitted on the validation predictions of the models trained without validation.
Outputs in /kaggle/working (test caches: (per_source, image_stem) per image):
  final_test_{v10,c512,hf}_ABC.pkl   sources A (Mask R-CNN 2048), B1280 (YOLO11m), C (RT-DETR)
  final_test_{v10,c512,hf}_L.pkl     YOLO11l under key B1280 -> rename to L1280 when merging
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
OUT = Path("/kaggle/working")


def find(name):
    return str(next(Path("/kaggle/input").rglob(name)))


MASKRCNN, YOLO_M, YOLO_L = find("maskrcnn_hires2048_all_epoch5.pt"), find("yolo11m_cls_all_best.pt"), find("yolo11l_cls_1280_all_best.pt")
RTDETR = find("rtdetr_cls_all_best.pt")
REFINERS = {"v10": find("refiner_v10_resnet34_full_best.pt"), "c512": find("refiner_v10_resnet34_c512_full_best.pt")}

s = (WORK / "sweep_ensemble_4way.py").read_text()
s = s.replace('RTDETR_CKPT = "kaggle_kernel_rtdetr_cls/output/rtdetr_cls_best.pt"', f'RTDETR_CKPT = "{RTDETR}"')
(WORK / "sweep_ensemble_4way.py").write_text(s)


def run(refiner, extra, args):
    cmd = [sys.executable, "-u", "sweep_ensemble_4way.py", "--refiner", refiner, "--maskrcnn", MASKRCNN] + extra + args
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


for tag, refiner, extra in [("v10", REFINERS["v10"], []), ("c512", REFINERS["c512"], []),
                            ("hf", REFINERS["v10"], ["--hflip"])]:
    run(refiner, extra, ["--build-test-cache", "--sources", "A", "B1280", "C", "--yolo", YOLO_M,
                         "--test-cache", str(OUT / f"final_test_{tag}_ABC.pkl")])
    run(refiner, extra, ["--build-test-cache", "--sources", "B1280", "--yolo", YOLO_L,
                         "--test-cache", str(OUT / f"final_test_{tag}_L.pkl")])
print("ALL DONE", flush=True)
