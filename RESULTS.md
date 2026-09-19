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
| **0.38** | YOLO11m-seg retrained with cls=1.5 (up from ultralytics' default 0.5), targeting a diagnosed root cause: several large, visually-obvious missed filaments had raw boxes with good localization (bbox-IoU 0.70-0.87 vs GT) but catastrophically low confidence (as low as 0.05, 7x below deployment threshold). Confidence re-tuned to the new peak (0.33): local val PQ 0.4171 -> 0.4252, TP 504 -> 522 -- **first real improvement past the 0.37 ceiling, current best** |
| 0.38 | Calibrated cross-architecture ensemble (Mask R-CNN cls-fixed + YOLO cls-fixed) with *true* cross-detector NMS dedup -- discard the lower-scored of two overlapping proposals wholesale instead of pooling and letting panoptic-paint fragment the loser. First of five ensemble strategies this session to actually beat solo detection locally: PQ 0.4252 -> 0.4361, TP 522 -> 574. Ties the 0.38 ceiling rather than beating it -- but a smooth, monotonic trend, not a knife-edge; refined further below |
| **0.39** | Refined the dedup_iou/accept grid around the tied result: the trend was still monotonically improving as dedup_iou decreased, turning over only below 0.01 (too aggressive -- starts merging genuinely-separate nearby filaments). True peak: dedup_iou=0.05 (flat plateau 0.03-0.1), accept=0.45. Local val PQ 0.4361 -> 0.4407, TP 574 -> 553. **Real score 0.38 -> 0.39, a genuine improvement past the ceiling -- current best, new recommended pipeline** (`predict_ensemble_dedup.py`) |
| (not submitted) | RT-DETR retrained with default `cls=0.5`/40 epochs as an untuned 3rd ensemble member: solo PQ ~0.40, but 3-way ensemble PQ=0.4383 < 2-way's 0.4407 -- diagnosed why: 58% of RT-DETR's well-localized boxes (bbox-IoU>0.5) scored below 0.5, median score just 0.031, far worse miscalibration than YOLO/Mask R-CNN's version of the same bug |
| 0.39 | Retrained RT-DETR with `cls=2.0` (4x default) + 60 epochs (up from 40) to fix the diagnosed miscalibration -- confirmed fixed: well-localized-box median score 0.031 -> 0.658, solo PQ ~0.40 -> **0.4296 (now the strongest solo detector of the three, beating YOLO's 0.4252 and Mask R-CNN's 0.4152)**. 3-way ensemble with the recalibrated RT-DETR: local val PQ 0.4407 -> **0.4418** across a genuinely flat plateau (dedup_iou 0.03-0.08 x accept 0.4-0.55 all identical) -- a real, non-fluke local improvement. **Real score: ties at 0.39** -- the +0.0011 local delta was too small to move the leaderboard's rounding. `predict_ensemble3_dedup.py` kept as documented (marginal, non-regressive) alternative; `predict_ensemble_dedup.py` (2-way) remains the simpler pipeline at the same real score |

See `README.md` for the architecture writeup and the full table of rejected
approaches with local numbers, and individual commit messages for the
reasoning behind each result.
