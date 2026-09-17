import json
from collections import Counter
from pathlib import Path

from PIL import Image

D = Path("data/MAGFiLO_1.0_Kaggle_2026")
ann_path = D / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

with open(ann_path, encoding="utf-8") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]
categories = {c["id"]: c["name"] for c in coco["categories"]}

print("=== info ===")
print(coco.get("info"))

print("\n=== categories ===")
print(categories)

print("\n=== counts ===")
print("images (annotated entries):", len(images))
print("annotations (filaments):", len(annotations))

img_files = set(i["file_name"] for i in images)
print("unique underlying image files:", len(img_files))

sizes = set((i["width"], i["height"]) for i in images)
print("image sizes seen in json:", sizes)

# filaments per image entry
per_image = Counter(a["image_id"] for a in annotations)
counts = list(per_image.values())
print("\nfilaments per annotated image-entry: min/mean/max =",
      min(counts), sum(counts) / len(counts), max(counts))

no_filament_images = [i["id"] for i in images if i["id"] not in per_image]
print("image-entries with zero filaments:", len(no_filament_images))

# class distribution
cat_counts = Counter(categories[a["category_id"]] for a in annotations)
print("\n=== class distribution (filaments) ===")
for name, cnt in cat_counts.most_common():
    print(f"  {name}: {cnt} ({100*cnt/len(annotations):.1f}%)")

# area / bbox stats
areas = [a["area"] for a in annotations]
areas.sort()
def pct(lst, p):
    return lst[int(len(lst) * p)]
print("\n=== filament area (px^2) ===")
print("min:", areas[0], "p25:", pct(areas, .25), "median:", pct(areas, .5),
      "p75:", pct(areas, .75), "max:", areas[-1])
print("mean:", sum(areas) / len(areas))
img_area = 2048 * 2048
print("mean area as % of image:", 100 * (sum(areas) / len(areas)) / img_area)

# polygon complexity (points per filament)
npoints = [len(a["segmentation"][0]) // 2 for a in annotations]
npoints.sort()
print("\n=== polygon vertex count ===")
print("min:", npoints[0], "median:", pct(npoints, .5), "max:", npoints[-1],
      "mean:", sum(npoints)/len(npoints))

# spine presence
has_spine = sum(1 for a in annotations if a.get("spine"))
print("\nannotations with non-empty spine:", has_spine, f"({100*has_spine/len(annotations):.1f}%)")

# repeated annotator batches (same underlying image, multiple annotators)
file_to_entries = Counter(i["file_name"] for i in images)
multi = {f: c for f, c in file_to_entries.items() if c > 1}
print("\nunderlying images with >1 annotator entry:", len(multi))
print("example:", list(multi.items())[:5])

# actual image file dims sample check (grayscale, size)
sample_files = list((D / "train" / "train_images").iterdir())[:5]
print("\n=== sample train image file check ===")
for p in sample_files:
    with Image.open(p) as im:
        print(p.name, im.size, im.mode)

sample_test = list((D / "test" / "test_images").iterdir())[:5]
print("\n=== sample test image file check ===")
for p in sample_test:
    with Image.open(p) as im:
        print(p.name, im.size, im.mode)

# do train_images file count match unique img_files referenced in json?
actual_train_files = set(p.name for p in (D / "train" / "train_images").iterdir())
print("\ntrain_images dir file count:", len(actual_train_files))
print("referenced in json but missing from dir:", len(img_files - actual_train_files))
print("in dir but not referenced in json:", len(actual_train_files - img_files))
