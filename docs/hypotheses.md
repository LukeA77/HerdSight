# Hypotheses

Written before each test is run. Each outcome is reported, whether the hypothesis held or not. An outcome is filled in only from a results file.

## Phase 1: baselines (written 2026-11-18, before any model was run)

All Phase 1 baselines are scored by grouped three-fold cross-validation over farm-date groups (D-26): every non-test image is predicted by a model that never saw its farm and day. The test farm (Other_farms) and the Shao et al. set are not touched until Phase 7.

*Edited 2026-11-18, before any model was run: "validation flights" and "calibration flights" became out-of-fold predictions when the split design changed (D-25, D-26). The predictions themselves are unchanged.*

| ID | Hypothesis | Prediction | How it is judged | Outcome |
|---|---|---|---|---|
| H1 | A detector fine-tuned on the training farms beats an open-vocabulary detector with no cattle training. | Fine-tuned centre-matched AP at least 0.2 higher, and per-image count mean absolute error (MAE) at most half. | Centre AP (mean over 0.50:0.95) and count MAE on out-of-fold images (the zero-shot image sample), with bootstrap 95% intervals over farm-date groups; the intervals must not overlap | **Not held.** Centre AP 0.51 vs 0.25 (gain met), but count MAE 0.77 vs 0.95 (not halved), and both pairs of intervals overlap (133 images) |
| H2 | Count errors grow with herd density. | Per-image MAE divided by the true count is at least twice as large for images with 20 or more animals as for images with 1 to 4. | Relative MAE per band of true count, with bootstrap intervals over farm-date groups | **Not held (reversed).** Relative MAE 0.12 for 20+ animals vs 0.42 for 1–4 |
| H3 | Conformal count intervals fitted on other folds hold on the held-out fold. | A nominal 90% interval contains the true count on 85% to 95% of out-of-fold images. | Coverage on out-of-fold images; also reported per herd-size band | **Held.** Coverage 0.93 (0.87–0.98); lowest on images with 20+ animals (0.78) |
| H4 | Detection is harder from 100 m than from 60 m. | Recall at the operating point at least 0.05 lower at 100 m. | Recall per altitude group, from the EXIF relative altitude, with bootstrap intervals over farm-date groups | **Not held (reversed).** Recall 0.75 at 60 m vs 0.88 at 100 m; the 60 m gap comes from the Derval farm, not height (failure review) |
| H5 | Empty pasture rarely produces false detections after fine-tuning. | Fewer than 0.05 false detections per image on out-of-fold images with no cattle. | False positives per empty out-of-fold image at the operating point | **Not held.** 0.33 false detections per empty image (0.07–0.73), mostly farmyard objects |
| H6 | Adding up per-image counts overcounts a flight, because overlapping images see the same animals. | The naive sum is more than 20% above the true number of animals on every flight tested. | Per-flight count against hand-linked identities (only if overlapping, geotagged flights exist; otherwise reported as not tested) | **Not held, narrowly.** Smallest overcount 17% over 10 verified flights; missed animals partly offset the double counts (with perfect detection every flight is above 20%) |

The operating point (score threshold) for each fold is the one that minimises count MAE on the other folds' out-of-fold predictions, fixed before that fold is scored.

## Phase 2: detector (written 2026-11-21, before any Phase 2 model was trained)

Scored on the same frozen three-fold cross-validation as Phase 1. "Reference" means the Phase 1 fine-tuned D-FINE S models (`results/baselines/dfine_scores.json`). The thresholds below are also recorded in `configs/phase2.yaml`, which is frozen before the first Phase 2 training run; each prediction is judged by a rule fixed in code before its run.

| ID | Hypothesis | Prediction | How it is judged | Outcome |
|---|---|---|---|---|
| H7 | Training for appearance (dark animals, shade, colour) and crowding helps most on the farm the model has not seen. | Derval recall at the operating point at least 0.15 higher than the reference, while pooled centre-matched recall falls by no more than 0.02. | Recall on Derval images and pooled centre-matched recall, out-of-fold, reference against the combined recipe (plan step G2) | pending |
| H8 | Hard negatives mined without fold leakage cut false detections on empty pasture. | False detections per empty image at most half the reference's. | False positives per empty out-of-fold image at the operating point | pending |
| H9 | The Phase 1 schedule (72 epochs) stopped short of what the model can learn. | The full 220-epoch schedule raises centre-matched recall by at least 0.03 on a training-side hold-out. | Plan step G0: one fold, 72 against 220 epochs, the same recipe and seed | pending |
| H10 | Oriented boxes help most where animals touch. | Recall on touching animals at least 0.05 higher than the best upright model's, at the operating point. | Recall on animals whose labelled box overlaps another, out-of-fold | pending |
| H11 | Count intervals fitted per density band fix the undercoverage on crowded images without becoming useless. | 90% coverage on images with 20 or more animals at least 0.85, with mean width no more than 1.5 times the width the pooled method gives on those images. | Coverage and width on the 20+ band, out-of-fold | pending |
| H12 | A larger model (D-FINE M) is more accurate than S with the same recipe. | Centre AP higher by more than the measured run-to-run spread, or by at least 0.03 where no repeat runs were made. | Paired difference over farm-date groups | pending |

## Phase 4: herd register and alerts (written 2026-11-21, before any alert rule was run)

Scored on simulated farms (`configs/sim_farms.yaml`, validation seed), never on field data. "Losses" are deaths and thefts; a loss is caught if an alert fires on its camp within 3 surveys. Each prediction is judged by a rule fixed in code before the run, at the 90% interval level unless stated, under the count error model measured in Phase 1.

| ID | Hypothesis | Prediction | How it is judged | Outcome |
|---|---|---|---|---|
| H13 | Alerting only when the whole interval is below the expected count removes most false alarms. | The interval rule has at least 5 times fewer false alarms per 100 surveys than the naive rule, and catches no more than 10 percentage points fewer losses of 5 or more animals. | False alarms per 100 surveys and share of losses of 5 or more caught, naive against interval rule | **Not held.** False alarms fell from 19.68 to 3.75 per 100 surveys (more than 5 times fewer), but the share of losses of 5 or more caught fell from 86% to 50%, more than the 10 points allowed |
| H14 | A recount before alerting removes most of the remaining false alarms at little cost in speed. | The recount rule has at most half the interval rule's false alarms per 100 surveys, and its mean surveys until a loss of 5 or more is caught is no more than 1 higher. | Same measures, interval against recount rule | **Held.** False alarms 3.75 → 1.51 per 100 surveys; mean surveys until caught 1.55 → 1.60 |
| H15 | Unrecorded moves cause most false alarms that survive the recount, and the farm-level check removes at least half of those. | The farm rule has at most half the recount rule's false alarms per 100 surveys on farms with unrecorded moves. | False alarms per 100 surveys, recount against farm rule, on farms whose unrecorded-move share is above 0.05 | **Not held.** On farms with unrecorded moves, false alarms 1.41 → 1.33 per 100 surveys; the farm-level check rarely finds a matching surplus |
