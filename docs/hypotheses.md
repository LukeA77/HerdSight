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
