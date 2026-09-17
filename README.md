# Solar Filament Segmentation Challenge 2026

Instance segmentation of solar filaments (thin, elongated dark absorption
features) in GONG H-alpha full-disk observations, for the
[Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026).
Scored by Panoptic Quality (PQ): `PQ = sum(IoU over matches) / (TP + 0.5*FP + 0.5*FN)`,
match = IoU > 0.5.

## Current best: PQ 0.37 (real leaderboard)

Two-stage detect-then-refine pipeline:

1. **Detection**: two independently-trained Mask R-CNN (`torchvision`)
   detectors -- one on the standard recipe, one with random-crop
   "tile" augmentation for scale diversity -- combined via an
   agreement-weighted candidate merge (`ensemble_diag.py`): a filament both
   detectors agree on needs only a modest confidence bar; one only a single
   detector found needs a much higher bar.
2. **Refinement**: each detection is cropped from the *original* full-res
   image (not the detector's blurry internal 28x28 mask head) and refined by
   a small U-Net (`train_refiner.py`) with an auxiliary output head
   supervised by the dataset's real, human-annotated spine (centerline)
   polyline -- real geometric signal, not a synthetic approximation.
3. **Test-time augmentation**: 4-view flip TTA with a topology-safe
   fallback (reject the flip-averaged prediction if it distorts the shape
   too much vs. the identity view).
4. Both detectors are refit on 100% of the labeled data (not just the 90%
   train split) after model selection was done on the held-out split --
   standard final-refit practice.

See `pq.py` for the exact local metric (matches the competition's) and
`diag.py` / `ensemble_diag.py` for the error-decomposition diagnostics used
to guide every change (total-miss vs. near-miss vs. spurious FP).

## What's been tried and rejected (with honest numbers)

Architectures/techniques that scored **worse** than the pipeline above, kept
for the record so they aren't retried blindly:

| Approach | Local PQ | Real PQ | Notes |
|---|---|---|---|
| Classical CV (threshold + connected components) | -- | 0.01 | |
| Mask R-CNN, single pass, no refiner | -- | 0.27 | |
| Plain U-Net (semantic) + connected components | 0.408 | 0.32 | can't split touching instances |
| Mask2Former (Swin-Tiny) | 0.373 | 0.28 | insufficient fine-tune budget on ~1k images |
| SAM (ViT-B decoder fine-tune, box-prompted) | 0.316 | -- | bottlenecked by the *same* detector recall problem |
| SAM2 (zero-shot automatic mask generation) | ~0.08-0.13 | 0.08 | genuinely promising per-point (0.717 IoU zero-shot!) but no good box/point source without a detector |
| Distance-transform + watershed (proposal-free) | 0.236 | 0.20 | confirmed the "detector recall" theory (total-miss *did* improve) but the watershed split and semantic head aren't good enough yet |

Tuning experiments that looked better **locally** but did not transfer to
the real leaderboard (documented so they are not re-tried without new
evidence):
- Loosening the ensemble's cross-detector candidate floor (0.50 -> 0.35):
  local PQ 0.407 -> 0.424-0.430, real PQ 0.37 -> 0.36 (confirmed twice)
- Training both ensemble detectors on 100% data (vs. one): local PQ
  unaffected, real PQ 0.37 -> 0.36 (ensemble diversity loss)
- More detector epochs (8 -> 12), more refiner crop context (1.8x -> 2.5x),
  more refiner resolution (256px -> 384px): all landed flat-to-negative
  locally, not submitted
- YOLO inference resolution raised further with confidence re-tuned at each
  step (imgsz 1792, conf 0.40, found via a resolution x confidence grid
  sweep on the val set): local PQ 0.411 -> 0.419, the largest local gain of
  the whole YOLO tuning line and a smooth single peak -- real PQ 0.37 ->
  0.36. **Refines the methodology lesson below**: it isn't enough for a
  sweep to look smooth: a *2-axis* grid over a 116-image val set overfits
  even when each individual axis looks clean, the same failure mode as the
  ensemble AGREE/UNIQUE grid search. A single new resolution value
  (imgsz 1536, confidence left at its already-validated 1280 setting) tied
  the real score instead of regressing it -- so the risk is specifically in
  jointly re-optimizing two axes at once, not in testing resolution itself.
- Cross-architecture ensemble (Mask R-CNN + YOLO) with a *calibrated* merge:
  fit an isotonic regression per detector (raw confidence -> P(true
  positive), on the val split) so both detectors' scores land on a
  comparable [0,1] scale, then pooled all candidates from both and painted
  by a single shared acceptance threshold. Principled fix for the "scores
  aren't on the same scale" problem that sank the earlier hand-tuned
  attempts below -- but still landed at local PQ 0.358 (best threshold),
  well short of either solo detector (~0.42-0.43). The calibration curves
  themselves looked reasonable and TP recall was genuinely the highest yet
  (634 vs ~505-520 for any solo detector, confirming the complementary
  detections are real), but pooling raw candidates from both detectors
  multiplies how often the *same* true filament gets several overlapping
  proposals; panoptic-paint's greedy highest-score-wins logic only keeps one
  winner per pixel region, and the leftover fragments from the losing
  proposals show up as extra near-miss/spurious predictions that eat the
  recall gain under PQ's symmetric FP/FN penalty. **This is now the third
  distinct ensemble-merge strategy (hand-tuned per-detector thresholds,
  grid-searched AGREE/UNIQUE thresholds, calibrated single threshold) to
  land well below solo detection** -- treat the cross-architecture ensemble
  direction as a closed line of investigation for this dataset/setup unless
  a genuinely different merge principle (e.g. de-duplicating overlapping
  cross-detector proposals *before* scoring, rather than after) is found.
- Ultralytics' built-in test-time augmentation (`model.predict(...,
  augment=True)`, flip + multi-scale, merged internally via NMS) on top of
  the validated YOLO11m config: PQ and TP came back *bit-for-bit identical*
  to `augment=False` (0.4171, TP=504) rather than merely similar. That's not
  "TTA doesn't help" so much as a sign the flag had no effect at all for
  this segmentation task/model combination in this ultralytics version --
  inconclusive, not a real negative result about TTA's potential.
- Self-training: every architecture/capacity change plateauing at the same
  real ceiling suggested a data-limited regime, so retrained YOLO11m on the
  real training set *plus* 116 test images pseudo-labeled by the current
  best pipeline at a strict conf>=0.75 floor (244 polygons total, ~11% more
  training images). Detector-level mask mAP50 was flat again (0.664 vs
  0.664), and the downstream PQ **regressed clearly**: 0.4171 -> 0.3931,
  TP 504 -> 420. Not submitted. Most likely explanation: even a strict
  confidence floor doesn't guarantee polygon *quality* -- pseudo-label
  boundaries come from the refiner's own (imperfect) crop-refine output
  rather than a human annotator, and any filament the base detector missed
  in a pseudo-labeled image becomes an implicit hard-negative during
  training, penalizing exactly the kind of borderline detection the model
  needs to get better at. Self-training from this pipeline's own outputs
  looks like it reinforces its existing blind spots rather than fixing them.
- Missed-filament root-cause analysis (`diag_missed_filaments.py`): the
  ~45% of GT filaments we miss entirely skew smaller/thinner/fainter than
  the ones we catch (median area 727 vs 1401, contrast 12.4 vs 16.5 gray
  levels) but the effect is modest, not a sharp cutoff -- and several large,
  visually-obvious misses don't fit that story at all. Manually inspecting
  one (a 27957px filament, clearly visible) found YOLO's raw candidate
  boxes had good localization (bbox-IoU 0.70-0.87 vs GT) but catastrophic
  confidence (as low as 0.05, 7x below our 0.35 operating threshold) --
  and feeding that exact box to the refiner produced a mask at IoU=0.75
  vs GT. This means for at least some misses, the detector *finds* the
  right region but its confidence head badly underrates it.
- Rescuing those candidates via post-hoc filtering, tried three ways, all
  failed to generalize despite the compelling single example above
  (`sweep_refiner_gate.py`, `sweep_contrast_rescue.py`):
  - Refiner's own mean-probability as the acceptance gate (replacing
    YOLO's score entirely): TP 504->624 but PQ crashed 0.418->0.21. The
    refiner is overconfident on out-of-distribution candidates too --
    confidence barely varies between real filaments and pure background
    junk once the YOLO floor is opened up, so it can't discriminate.
  - Refiner confidence + a tiny YOLO floor (>=0.05) combined: same
    failure, PQ~0.27.
  - Local contrast (the measured, real property that *did* separate
    missed from caught filaments in the root-cause analysis) as a rescue
    filter for candidates YOLO scored in [0.02, 0.35): even the strictest
    threshold tested only recovered +4 TP while still net-negative on PQ
    (0.414 vs 0.418 baseline); looser thresholds traded away far more
    precision than they gained in recall.
  **Conclusion**: the missed-filament pool isn't cheaply separable from
  noise using any single post-hoc signal tried so far. The compelling
  single example was real but not representative -- most of the rescue
  pool actually is junk. The likely fix, if any, is upstream: better
  confidence calibration during YOLO training itself (e.g. loss
  reweighting), not smarter filtering of what it already outputs.

## Known leaderboard contamination

**Scores above ~0.5 on the public leaderboard are not legitimate models.**
77% of the test set is byte-identical to images already publicly released
(with ground truth) in the source MAGFiLO 1.0 dataset. The competition host
has confirmed the real evaluation is a separate quantitative + qualitative
rubric, not the public leaderboard, and that "any PQ score greater than 0.3
is of great value" to them. This repo never uses that leaked overlap.

## Repo layout

- `dataset.py`, `crop_dataset.py` -- COCO annotation loading, leak-safe
  train/val split (grouped by file, not COCO image id), per-instance crop
  cache building
- `train.py`, `train_refiner.py` -- detector and refiner training
- `predict_refined.py`, `ensemble_diag.py`, `predict_ensemble2.py` -- single
  and ensemble inference (`ensemble_diag.py` uses only the held-out-safe
  90%-split checkpoints for honest local validation; `predict_ensemble2.py`
  is the actual submission pipeline)
- `pq.py`, `diag.py` -- local metric + error decomposition
- `validate_submission.py` -- submission format / disjointness checker
- Everything else (`train_mask2former.py`, `train_sam.py`,
  `predict_sam2_auto.py`, `train_watershed.py`, ...) -- the rejected
  architectures above, kept for the record
