"""Native-2048 Mask R-CNN (the deployed A recipe) retrained with the augmentations
it never had -- transpose (completing the 8 dihedral orientations with the two
flips) and on-disk gamma/contrast/brightness jitter -- for 8 epochs, then its
val/test candidates refined with v10 (source A only), using the repo code.
Outputs: aug_val_A.pkl, aug_test_A.pkl, maskrcnn_hires2048_aug_epoch7.pt
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
link = WORK / "data" / "MAGFiLO_1.0_Kaggle_2026"
if not link.exists():
    link.symlink_to(next(Path("/kaggle/input").rglob("MAGFiLO_1.0_Kaggle_2026")))
os.chdir(WORK)
OUT = Path("/kaggle/working")


def run(cmd):
    print(" ".join(cmd), flush=True)
    subprocess.run([sys.executable, "-u"] + cmd, check=True)


run(["train_maskrcnn_hires.py", "--min-size", "2048", "--max-size", "2048", "--epochs", "8",
     "--tag", "_aug", "--seed", "4", "--rot90", "--photometric"])
ckpt = WORK / "checkpoints" / "maskrcnn_hires2048_aug_epoch7.pt"
shutil.copy(ckpt, OUT / ckpt.name)
common = ["sweep_ensemble_4way.py", "--sources", "A", "--maskrcnn", str(ckpt),
          "--refiner", str(SRC / "refiner_v10_resnet34_best.pt"), "--yolo", str(SRC / "yolo11m_cls_best.pt")]
run(common + ["--cache", str(OUT / "aug_val_A.pkl")])
run(common + ["--build-test-cache", "--test-cache", str(OUT / "aug_test_A.pkl")])
shutil.rmtree(WORK / "checkpoints", ignore_errors=True)
print("ALL DONE", flush=True)
