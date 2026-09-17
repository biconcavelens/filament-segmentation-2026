import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

D = Path("data/MAGFiLO_1.0_Kaggle_2026")
ann_path = D / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

with open(ann_path, encoding="utf-8") as f:
    coco = json.load(f)

per_image = {}
for a in coco["annotations"]:
    per_image.setdefault(a["image_id"], []).append(a)

# pick the image entry with the most filaments for a rich example
img_entry = max(coco["images"], key=lambda i: len(per_image.get(i["id"], [])))
anns = per_image[img_entry["id"]]
print("chosen image_id:", img_entry["id"], "file:", img_entry["file_name"], "n_filaments:", len(anns))

img_path = D / "train" / "train_images" / img_entry["file_name"]
base = Image.open(img_path).convert("RGB")
overlay = base.copy()
draw = ImageDraw.Draw(overlay, "RGBA")

colors = {
    "Left": (255, 80, 80, 110),
    "Right": (80, 160, 255, 110),
    "Unidentifiable": (255, 220, 60, 110),
    "Ambiguous": (160, 80, 255, 110),
}
cat_names = {c["id"]: c["name"] for c in coco["categories"]}

for a in anns:
    poly = a["segmentation"][0]
    pts = list(zip(poly[0::2], poly[1::2]))
    name = cat_names[a["category_id"]]
    draw.polygon(pts, fill=colors[name], outline=(255, 255, 255, 200))
    spine = a.get("spine")
    if spine:
        spts = list(zip(spine[0::2], spine[1::2]))
        draw.line(spts, fill=(0, 0, 0, 255), width=3)

combo = Image.new("RGB", (base.width * 2, base.height))
combo.paste(base, (0, 0))
combo.paste(overlay, (base.width, 0))
combo = combo.resize((combo.width // 2, combo.height // 2))
out = Path("scratch"); out.mkdir(exist_ok=True)
out_path = out / "sample_overlay.png"
combo.save(out_path)
print("saved:", out_path)
