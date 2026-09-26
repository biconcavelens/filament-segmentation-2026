"""YOLO11m-seg (cls=1.5, the deployed recipe) trained at native 2048px.

YOLO@2048 *inference* on the 1280-trained model helped val every time and
scored 0.38 on all three submissions that used it, while Mask R-CNN
*trained* at 2048 held at 0.39 -- so the likely problem is the train/
inference resolution mismatch, not resolution. This removes the mismatch.
Same recipe as kaggle_kernel_train_cls (epochs 40, patience 15, seed 0),
only imgsz changes; batch shrinks to fit 8GB.

    python train_yolo_hires.py --imgsz 2048 --batch 2
    python train_yolo_hires.py --imgsz 2048 --batch 2 --fraction 0.05 --epochs 1   # smoke test
"""
import argparse

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--imgsz", type=int, default=2048)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    name = f"yolo11m_cls_{args.imgsz}"
    if args.resume:
        YOLO(f"runs/segment/yolo_runs/{name}/weights/last.pt").train(resume=True)  # ultralytics nests under runs/segment
        return
    YOLO("yolo11m-seg.pt").train(
        data="yolo_data/data.yaml", epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
        patience=15, seed=0, deterministic=True, cls=1.5, fraction=args.fraction,
        project="yolo_runs", name=name, exist_ok=True, workers=2, amp=True)


if __name__ == "__main__":
    main()
