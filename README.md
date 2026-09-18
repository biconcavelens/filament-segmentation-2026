# Solar Filament Segmentation Challenge 2026

Instance segmentation of solar filaments (thin, elongated dark absorption
features) in GONG H-alpha full-disk observations, for the
[Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026).
Scored by Panoptic Quality (PQ): `PQ = sum(IoU over matches) / (TP + 0.5*FP + 0.5*FN)`,
match = IoU > 0.5.

## Current best: PQ 0.38 (real leaderboard)

Two-stage detect-then-refine pipeline:

1. **Detection**: YOLO11m-seg (`kaggle_kernel_train_cls/train_yolo_cls.py`),
   trained with a raised classification-loss weight (`cls=1.5`, up from
   ultralytics' default 0.5) to fix a diagnosed confidence-miscalibration
   problem: `diag_missed_filaments.py` found several large, visually-obvious
   missed filaments where YOLO's raw box was well-localized (bbox-IoU
   0.70-0.87 vs GT) but scored as low as 0.05, 7x below the deployment
   threshold. Inference at imgsz=1280, conf=0.33 (re-tuned for this
   checkpoint via a cached single-axis sweep).
2. **Refinement**: each detection is cropped from the *original* full-res
   image (not the detector's blurry internal mask head) and refined by
   a small U-Net (`train_refiner.py`) with an auxiliary output head
   supervised by the dataset's real, human-annotated spine (centerline)
   polyline -- real geometric signal, not a synthetic approximation.
3. **Test-time augmentation**: 4-view flip TTA with a topology-safe
   fallback (reject the flip-averaged prediction if it distorts the shape
   too much vs. the identity view).

An earlier two-detector Mask R-CNN ensemble (agreement-weighted candidate
merge, `ensemble_diag.py`) and solo YOLO11s/YOLO11m without the cls-weight
fix all independently reached real PQ 0.37 -- see RESULTS.md for the full
progression to 0.38.

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
- **Fourth rescue-filter attempt, same story**: `sweep_multifeature_rescue.py`
  combined all the weak signals above (raw score, contrast, area,
  elongation, thickness) into a small logistic regression classifier
  instead of a single threshold, fit on a genuinely separate pool this time
  (candidates from 250 *train* images, not val, to avoid the fit/eval
  circularity of the earlier isotonic-calibration attempts). Best result:
  accept_prob>=0.9, PQ=0.4235 (+4 TP), essentially identical to the
  contrast-only attempt's best result and still below the 0.4252 baseline.
  **This closes the post-hoc rescue-filtering investigation for good**: four
  different feature combinations (refiner confidence alone, refiner
  confidence + floor, local contrast alone, and now all of the above
  combined via a classifier) all converge to the same story -- loosen the
  gate and precision craters, tighten it to where it's safe and you barely
  recover anything. The rescue-zone candidate pool (YOLO score 0.02-0.33)
  genuinely does not contain a cheaply-extractable signal separating real
  misses from noise; the only lever that has actually worked is fixing the
  upstream model (the cls-weight training fix, real PQ 0.37 -> 0.38).
- **Follow-up, and this one worked**: retrained YOLO11m with `cls=1.5`
  (ultralytics' classification-loss weight, up from the default 0.5) to
  push exactly the upstream fix predicted above. Local val PQ 0.4171 ->
  0.4252 (TP 504 -> 522) at a re-tuned conf=0.33, a smooth single-peaked
  sweep. **Real score: 0.37 -> 0.38, the first genuine improvement past the
  plateau all session.** Confirms the root-cause diagnosis was correct: the
  bottleneck for a meaningful chunk of the missed filaments really was
  confidence-head undertraining, not detector capacity, resolution, or
  ensemble diversity -- all of which were tried first and all plateaued.
- Pushed the same lever further to see if the trend continues: `cls=3.0`
  (double the winning 1.5). It doesn't -- detector-level mask mAP50 dropped
  to 0.647 (worse than even the *original* uncalibrated 0.664), and local
  val PQ peaked at only 0.4104, below both cls=1.5 (0.4252) and the
  uncalibrated baseline (0.4171). Not submitted. `cls=1.5` sits near the
  useful optimum for this hyperparameter -- overweighting classification
  loss far enough starts trading away box/segmentation quality instead of
  fixing confidence calibration for free.
- Applied the same fix to Mask R-CNN (the *other* detector family, still at
  real PQ 0.37): a diagnostic mirroring `diag_missed_filaments.py` found
  the identical bug, worse in relative terms -- 231/908 GT filaments (25%)
  had a well-localized box (bbox-IoU>0.5) but scored below the 0.80
  deployment threshold, median score only 0.386. Retrained with the RoI
  head's `loss_classifier` reweighted 3x (torchvision doesn't expose a
  `cls=` kwarg like ultralytics, so this reweights the term directly in the
  training loop) and re-swept the acceptance threshold: smooth single peak
  at thresh=0.80, local val PQ 0.4152. Solid on its own but below the
  cls-fixed YOLO (0.4252); training was still improving at the last epoch
  here too, so there's likely more in this checkpoint with a longer run.
  Not submitted solo (wouldn't beat the current best), but confirms the
  confidence-miscalibration bug is a property of *this problem* (thin,
  variably-faint filaments), not one specific architecture's quirk.
- **Retried the calibrated cross-architecture ensemble** with both
  detectors now individually confidence-fixed (previous calibrated-ensemble
  attempt used the *old*, miscalibrated Mask R-CNN and YOLO11m). If the
  earlier failure really was about scores not being comparable across
  architectures, fixing calibration on both sides first should have helped.
  It didn't: best ensemble PQ=0.3492, still well below *either* solo
  detector (0.4252 YOLO, 0.4152 Mask R-CNN). **This is the fourth distinct
  ensemble-merge strategy to fail**, and calibration quality clearly isn't
  the bottleneck (both inputs were reasonably calibrated this time). The
  real problem is more likely the one flagged after the first calibrated
  attempt: pooling raw candidates from two architectures means the same
  true filament gets several overlapping proposals, and panoptic-paint's
  greedy pixel-claiming can't resolve that the way a proper cross-detector
  NMS/dedup would. Cross-architecture ensembling looks structurally closed
  for this pipeline, not just under-calibrated.
- Tried extending the winning cls=1.5 run further: its loss curve was still
  improving at epoch 40 (val/cls_loss still falling, mAP50(M) still
  climbing), so continued training for 30 more epochs starting from the
  epoch-40 weights. True ultralytics `resume=True` (which preserves
  optimizer/LR-scheduler state exactly) needs the checkpoint's own recorded
  save_dir to exist, which breaks across a fresh Kaggle kernel filesystem --
  so this used a fresh training call seeded from the good weights instead
  (its own new LR schedule/warmup). Result: peaked at local val PQ=0.4170,
  essentially flat-to-slightly-worse than the original 40-epoch run's
  0.4252. A fresh warmup restarting from already-converged weights doesn't
  cleanly extend the original trajectory -- it behaves like an independent
  run that happens to start from a good init, not a continuation. Not
  submitted. A true state-preserving resume might still help; the
  cross-machine save_dir issue would need solving first.
- Every experiment above targeted the *detector* stage. `sweep_refiner_spine_weight.py`
  targets the refiner instead -- the component that determines mask
  boundary quality once something's already detected -- by sweeping the
  auxiliary spine-centerline loss weight (`spine_w`, default 0.3) from
  0.15 to 0.6. Cheap to test locally (small crop-based U-Net, ~40min for
  all 4 variants). Caveat worth flagging: the raw training val_loss isn't
  comparable across spine_w values, since it mechanically reweights the
  loss definition itself (a higher spine_w shifts weight toward the
  easier-to-fit spine term, lowering the number without necessarily
  improving mask quality) -- so all 4 checkpoints were run through the
  actual detect+refine+PQ pipeline instead. Result: flat, PQ 0.4225-0.4259
  across the whole range, no meaningful difference. Confirms the refiner
  isn't the bottleneck -- consistent with the session-long finding that
  missed detections (recall), not boundary precision of what's already
  caught, account for nearly all the remaining PQ gap.

- Combined the cls=1.5 fix with a second, genuinely untried lever: cleaner
  training labels. `dataset.py`'s own docstring flags the mechanism --
  "42% of images have 2-3 annotators who disagree on *which* filaments to
  mark, so training on all entries teaches the detector to sometimes skip
  real filaments" -- a very plausible contributor to the same confidence-
  miscalibration bug the cls=1.5 fix addressed. `dataset.py` already
  supports `train_labels="complete"` (keep only the richest annotator
  entry per training image) but it had never been tried for YOLO, or
  combined with the cls-weight fix. Result: mAP50(M) 0.649 (vs 0.671 for
  "all" labels), local val PQ peaked at 0.4188 (vs 0.4252) -- worse, not
  better. Root cause: "complete" filtering collapsed 1038 training entries
  down to 637 unique images, a ~39% cut in training volume. The lost data
  volume outweighed whatever label-quality benefit existed. Not submitted.
- **Correction to "cross-architecture ensembling looks structurally
  closed" above**: the fourth attempt correctly diagnosed the mechanism
  (duplicate overlapping cross-detector proposals) but never actually
  fixed it -- it still pooled candidates and let panoptic-paint's greedy
  pixel-claiming handle overlaps implicitly, the same thing every prior
  attempt did. A fifth attempt (`sweep_ensemble_true_dedup.py`) implements
  the actual fix: explicit cross-detector NMS *before* painting -- sort
  calibrated candidates by score, greedily accept, and discard (not
  fragment) anything overlapping an already-accepted candidate above an
  IoU threshold. This is the first of five ensemble strategies to beat
  solo detection: local val PQ 0.4361 (dedup_iou=0.3, accept=0.4) vs
  0.4252 YOLO solo / 0.4152 Mask R-CNN solo, TP 574 vs 522 -- the largest
  local gain of the whole post-checkpoint exploration phase, and a smooth
  plateau across nearby grid cells (0.43-0.436) rather than a knife-edge
  spike. Submitted (`predict_ensemble_dedup.py`): **real score 0.38,
  ties rather than beats the current best.** Kept as a documented,
  genuinely-better-locally alternative -- the tie (not a regression) is
  itself useful signal that the fix is real, just not yet large enough to
  cross into a higher score bucket on this ~180-image test set. The
  simpler solo YOLO pipeline remains the primary recommendation for the
  final report given the tie and substantially lower complexity (one
  detector, no calibration-fitting step, easier to document and
  reproduce) -- see the Open-Access Policy / code-quality scoring in the
  competition rubric.

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
