"""Fine-tune torchvision's Mask R-CNN for single-class filament segmentation.

Usage:
    python train.py --epochs 8
"""
import argparse
import time
from pathlib import Path

import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from dataset import FilamentDataset, collate_fn, train_val_split

CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

# ponytail: min_size/max_size cut resolution to fit an 8GB laptop GPU; a
# beefier GPU could train closer to native 2048px for better barb recall.
MIN_SIZE = 800
MAX_SIZE = 1333


def build_model(num_classes=2):
    model = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1,
        min_size=MIN_SIZE, max_size=MAX_SIZE,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    return model


V3_SIZE = 1024


def build_model_v3(num_classes=2):
    """Reference-notebook config: 16px anchors (default smallest is 32, but a
    filament's short side is ~14px at this scale), 1024 input, and 2x mask RoI
    resolution so the coarse mask head sees elongated shapes less blurrily."""
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.ops import MultiScaleRoIAlign

    model = build_model(num_classes)
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)), aspect_ratios=((0.5, 1.0, 2.0),) * 5)
    model.roi_heads.mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=28, sampling_ratio=2)
    model.transform.min_size, model.transform.max_size = (V3_SIZE,), V3_SIZE
    return model


def build_from_checkpoint(state: dict, num_classes=2):
    return build_model_v3(num_classes) if state.get("arch") == "v3" else build_model(num_classes)


@torch.no_grad()
def evaluate_loss(model, loader, device):
    losses = []
    for images, targets in loader:
        images = [i.to(device) for i in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        model.train()  # torchvision only returns losses in train mode
        loss_dict = model(images, targets)
        losses.append(sum(loss_dict.values()).item())
    return sum(losses) / len(losses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--tag", type=str, default="",
                    help="checkpoint filename prefix, e.g. 'v2' -> maskrcnn_v2_epochN.pt")
    p.add_argument("--v3", action="store_true", help="16px anchors, 1024 input, 2x mask RoI")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--labels", choices=["all", "complete"], default="all",
                    help="'complete' trains on only the most complete annotator entry per image")
    p.add_argument("--copy-paste", action="store_true",
                    help="paste real filaments from other images onto each training image")
    p.add_argument("--tile-aug", action="store_true",
                    help="train on random 1152px tile crops half the time, so the detector is "
                         "calibrated for tiled (higher-effective-resolution) inference too")
    p.add_argument("--val-frac", type=float, default=0.1,
                    help="0 trains on all available labeled images (final-model refit after "
                         "model selection is done using the normal held-out split)")
    args = p.parse_args()
    ckpt_prefix = f"maskrcnn_{args.tag}" if args.tag else "maskrcnn"
    arch = "v3" if args.v3 else "v1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "arch:", arch, "labels:", args.labels, "val_frac:", args.val_frac)

    train_entries, val_entries, per_image = train_val_split(
        val_frac=args.val_frac, seed=0, train_labels=args.labels)
    train_ds = FilamentDataset(train_entries, per_image, augment=True,
                                copy_paste=args.copy_paste, tile_aug=args.tile_aug)
    val_ds = FilamentDataset(val_entries, per_image)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=2
    )

    model = (build_model_v3 if args.v3 else build_model)(num_classes=2).to(device)
    start_epoch = 0

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    milestones = sorted({max(1, round(args.epochs * 0.7)), max(2, round(args.epochs * 0.9))})
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        start_epoch = state["epoch"] + 1
        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            scheduler.load_state_dict(state["scheduler"])
        else:
            print("  warning: checkpoint has no optimizer/scaler/scheduler state "
                  "(pre-fix checkpoint) -- resuming with fresh momentum")
            for _ in range(start_epoch):
                scheduler.step()
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        for i, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()

            if (i + 1) % 50 == 0:
                print(f"  epoch {epoch} step {i+1}/{len(train_loader)} "
                      f"loss={running/(i+1):.3f}")
            if (i + 1) % 200 == 0:
                torch.cuda.empty_cache()
                time.sleep(8)

        scheduler.step()
        val_loss = evaluate_loss(model, val_loader, device) if len(val_loader) else float("nan")
        dt = time.time() - t0
        print(f"epoch {epoch}: train_loss={running/len(train_loader):.3f} "
              f"val_loss={val_loss:.3f} ({dt:.0f}s)")

        ckpt = CKPT_DIR / f"{ckpt_prefix}_epoch{epoch}.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch, "arch": arch,
                    "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "scheduler": scheduler.state_dict()}, ckpt)
        print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
