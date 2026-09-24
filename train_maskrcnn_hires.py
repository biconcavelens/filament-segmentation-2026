"""Mask R-CNN (cls-weighted, the deployed recipe) trained at a higher input
resolution. Every Mask R-CNN so far ran at min/max 800/1333 -- a leftover of
an 8GB-GPU memory limit (train.py's ponytail note) -- so 2048px images are
downscaled ~1.5-2.5x before the backbone sees them. The dominant remaining
error is truncated masks on long, faint filaments (diag_near_miss*.py); the
faint tails are exactly the detail downscaling removes.

Same recipe as kaggle_kernel_maskrcnn_cls (CLS_WEIGHT=3, SGD, AMP,
MultiStepLR at 70%/90%), only the input size changes. The size is stored in
the checkpoint and applied by train.build_from_checkpoint at inference.

    python train_maskrcnn_hires.py --min-size 1333 --max-size 2048 --epochs 6
"""
import argparse
import time

import torch

from dataset import FilamentDataset, collate_fn, train_val_split
from train import build_model, CKPT_DIR


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-size", type=int, default=1333)
    p.add_argument("--max-size", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--cls-weight", type=float, default=3.0)
    p.add_argument("--max-steps", type=int, default=0, help="stop each epoch early (smoke test)")
    args = p.parse_args()
    lr = 0.005 * args.batch_size / 2  # the kernel used 0.005 at batch 2; linear scaling
    name = f"maskrcnn_hires{args.min_size}"

    device = torch.device("cuda")
    train_entries, _, per_image = train_val_split(val_frac=0.1, seed=0)
    loader = torch.utils.data.DataLoader(
        FilamentDataset(train_entries, per_image, augment=True),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)

    model = build_model(num_classes=2)
    model.transform.min_size, model.transform.max_size = (args.min_size,), args.max_size
    model.to(device)
    params = [q for q in model.parameters() if q.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scaler = torch.amp.GradScaler("cuda")
    milestones = sorted({max(1, round(args.epochs * 0.7)), max(2, round(args.epochs * 0.9))})
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    print(f"{name}: size {args.min_size}/{args.max_size} bs={args.batch_size} lr={lr} "
          f"cls_weight={args.cls_weight} steps/epoch={len(loader)}", flush=True)

    for epoch in range(args.epochs):
        model.train()
        t0, running = time.time(), 0.0
        for i, (images, targets) in enumerate(loader):
            if args.max_steps and i >= args.max_steps:
                break
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                ld = model(images, targets)
                loss = (ld["loss_box_reg"] + ld["loss_mask"] + ld["loss_objectness"] +
                        ld["loss_rpn_box_reg"] + args.cls_weight * ld["loss_classifier"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            if (i + 1) % 100 == 0:
                print(f"  epoch {epoch} step {i + 1}/{len(loader)} loss={running / (i + 1):.3f} "
                      f"{(time.time() - t0) / (i + 1):.2f}s/step "
                      f"peak_mem={torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
        scheduler.step()
        print(f"epoch {epoch}: loss={running / max(i, 1):.3f} ({time.time() - t0:.0f}s)", flush=True)
        torch.save({"model": model.state_dict(), "epoch": epoch, "arch": None,
                    "min_size": args.min_size, "max_size": args.max_size},
                   CKPT_DIR / f"{name}_epoch{epoch}.pt")


if __name__ == "__main__":
    main()
