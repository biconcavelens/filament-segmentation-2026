"""Convert our COCO-style filament annotations into YOLO segmentation format
(one .txt per image: `class_id x1 y1 x2 y2 ...` normalized polygon per
instance). Multi-annotator entries share the same underlying image file but
have different annotation sets, so each COCO entry (not unique file) becomes
one YOLO sample, named by its unique entry id to avoid collisions -- same
convention FilamentDataset already uses for Mask R-CNN training.
"""
import shutil
from pathlib import Path

from dataset import IMG_DIR, H, W, train_val_split

YOLO_ROOT = Path("yolo_data")


def _safe_name(entry_id: str) -> str:
    return entry_id.replace("/", "_").replace("\\", "_")


def build_split(entries: list[dict], per_image: dict, split: str):
    img_dir = YOLO_ROOT / "images" / split
    lbl_dir = YOLO_ROOT / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for e in entries:
        anns = per_image.get(e["id"], [])
        name = _safe_name(e["id"])
        src_img = IMG_DIR / e["file_name"]
        dst_img = img_dir / f"{name}.jpeg"
        if not dst_img.exists():
            shutil.copy(src_img, dst_img)

        lines = []
        for a in anns:
            for poly in a["segmentation"]:
                pts = poly
                if len(pts) < 6:  # need at least 3 points
                    continue
                norm = []
                for i in range(0, len(pts), 2):
                    x, y = pts[i] / W, pts[i + 1] / H
                    norm.append(f"{x:.6f} {y:.6f}")
                lines.append("0 " + " ".join(norm))
        (lbl_dir / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_written += 1
    return n_written


def build_dataset():
    train_entries, val_entries, per_image = train_val_split(val_frac=0.1, seed=0)
    n_train = build_split(train_entries, per_image, "train")
    n_val = build_split(val_entries, per_image, "val")
    print(f"train: {n_train} images, val: {n_val} images")

    yaml_content = f"""path: {YOLO_ROOT.resolve()}
train: images/train
val: images/val
names:
  0: filament
"""
    (YOLO_ROOT / "data.yaml").write_text(yaml_content, encoding="utf-8")
    print(f"wrote {YOLO_ROOT / 'data.yaml'}")


if __name__ == "__main__":
    build_dataset()
