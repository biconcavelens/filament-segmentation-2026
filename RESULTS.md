# Submission history (real Kaggle leaderboard)

| Score | Approach |
|---|---|
| 0.01 | Classical CV baseline (threshold + connected components) |
| 0.27 | Mask R-CNN, single pass, no refiner |
| 0.30-0.34 | Mask R-CNN + crop-refine U-Net, various threshold sweeps |
| 0.28 | Mask2Former (Swin-Tiny) |
| 0.32 | Plain U-Net (semantic) + connected components |
| 0.34 | Mask R-CNN + crop-refine U-Net + 4-view flip TTA |
| 0.35 | + NMS threshold tuned (0.5 -> 0.3) |
| 0.36 | + two-detector agreement-weighted ensemble, mixed-scale training |
| **0.37** | + full-data detector refit, spine-supervised refiner (**current best**) |
| 0.36 | Full-data refit of *both* ensemble detectors (regression: ensemble diversity loss, confirmed twice) |
| 0.08 | SAM2 zero-shot automatic mask generation (no fine-tuning, no detector) |
| 0.20 | Distance-transform + watershed (proposal-free) |
| 0.36 | Loosened ensemble candidate floor (0.50 -> 0.35), locally better, real regression (confirmed twice) |
| 0.36 | Solo YOLO11s-seg detector + proven crop-refine refiner (matches 0.37 pipeline, different architecture) |
| **0.37** | + confidence threshold tuned (0.25 -> 0.35 via single-axis cached sweep) -- **ties current best, single detector** |
| 0.37 | + inference resolution raised (imgsz 1280 -> 1536), local val PQ 0.4114 -> 0.4131, TP 495 -> 519 -- **ties best, real gain too small to move the leaderboard score** |
| 0.36 | + imgsz raised further to 1792 with confidence re-tuned (0.40), found via a resolution x confidence grid sweep: local val PQ 0.4114 -> 0.4187 (largest local gain of the whole YOLO tuning line, smooth single peak) -- **real regression despite the strongest local signal yet; reverted to imgsz=1280/conf=0.35 (0.37) as current best** |
| 0.37 | YOLO11m-seg (up from 11s) trained from scratch on Kaggle GPU, same validated imgsz=1280/conf=0.35 config: local val PQ 0.4114 -> 0.4171, TP 495 -> 504 -- **ties best, third genuine single-axis local win in a row (confidence tune, this) that caps at 0.37; solo-detector family looks plateaued at this ceiling** |
| (not submitted) | Self-training: retrained YOLO11m on real train + 116 strictly-filtered pseudo-labeled test images (conf>=0.75, 244 polygons). Local val PQ 0.4171 -> **0.3931, TP 504 -> 420 -- clear regression**, not submitted |

See `README.md` for the architecture writeup and the full table of rejected
approaches with local numbers, and individual commit messages for the
reasoning behind each result.
