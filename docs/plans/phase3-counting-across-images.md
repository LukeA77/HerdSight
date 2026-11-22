# Phase 3 plan: geolocation, duplicate removal and count intervals

Status: agreed by the owner (2026-11-22), after self-review and revision. Runs alongside Phase 2 (G2 is training on a rented GPU) and Phase 5; it uses the Phase 1 detector's out-of-fold detections now and is re-run with the Phase 2 model when that is chosen.

## Goal
Count each animal once per flight, even when it appears in several overlapping stills, and give the count with a calibrated range, e.g. "143 (141–145 at 90%)" (D-9).

Exit (brief §12): **S3 on validation; parity tests pass.** S3 is the mean absolute relative error of the flight total on the owner-verified flights, target at most 0.15 (`configs/specs.yaml`, D-31), plus the same measure on simulated flights. Every piece of maths that exists in both Python and C++ has a parity test.

## Where Phase 1 left it
- **Method (D-27, D-30):** overlapping stills are registered from image content (a homography fitted by RANSAC to feature matches), detections are paired one-to-one in registered pixels (geotag projection as fallback), and pairs are joined greedily, shortest first, never putting two detections of one still in one animal.
- **Result** (`results/baselines/dfine_s3.json`, Phase 1 detector, out of fold): on the 10 owner-verified flights the mean absolute relative error is 0.26 (geotag only 0.35; no duplicate removal 2.70).
- **Where it fails:** flights with many overlapping pairs over a small herd (Mauron 20231107 flight 065: 25 animals, 460 overlapping pairs, +76%; flight 025: +21%), where small linking errors pile up, and flights with missed animals (Derval: −33%, mostly detection misses on the farm the model never saw, C1).
- **Truth:** 10 flights verified by the owner from candidate links built on the labels (`results/data/s3_truth.json`); 10 more flights had unverified candidate counts only. The owner verified those 10 too on 2026-11-22 (review round 3, `scripts/make_link_review.py --round 3`), so all 20 flights now have complete verified counts; the first 10 are unchanged.
- **C++ core:** `core/` (C++20, Eigen, GoogleTest under sanitizers, pybind11 module `herdsight._core`, parity tests) holds only the frame rotations so far.

## Approach

### 1. Split the error before changing anything
Run the Phase 1 method with the labels as detections ("perfect detection") and with the model's detections, on every flight. The first gives the linking error alone; the difference is what detection adds. Break both down per flight by overlap, time between passes, herd density and registration success. This decides where the effort goes, as C1 did for Phase 2.

### 2. Evaluation design (fixed before any tuning)
- **More verified flights.** The 10 unverified flights are sent to the owner for the same review (about 1–2 hours), so 20 flights have verified counts, with at least five in each fold.
- **Everything learned is fitted out of fold.** Anything fitted to data (the same-animal model in step 4, gates, weights) is fitted on the flights of two folds and applied to the third, the same farm-date folds as the detector (D-26). S3 is then the mean over all verified flights, each scored by a model that never saw its fold. No setting is chosen by looking at a flight's S3 score.
- **Simulation** (Phase 5, `python/herdsight/sim/`): flights with exact truth, moving animals and the fitted detection, geotag and registration error models. Validation seeds are used during the work; test seeds run once in Phase 7.
- **Uncertainty:** a bootstrap interval over flights for every real-flight figure, and paired comparisons (the same flights, two methods) for every claim that one method is better.

### 3. Ground positions with honest uncertainty
Each detection gets a ground position and a covariance from the camera geometry and the geotag error (`geo/ground_cov.py`, the per-flight geotag model fitted in Phase 5), and each registered pair a pixel uncertainty from its RANSAC fit. Registration failures fall back to the geotag with its larger covariance, as now, and are counted.

### 4. Evidence that two detections are one animal
For every pair of detections in overlapping stills:
- **position:** distance in registered pixels or on the ground, scaled by the combined uncertainty;
- **movement allowance:** how far a cow can plausibly move in the time between the two stills, from the measured grazing drift and walking speeds (`results/data/herd_motion.json`, `configs/sim_worlds.yaml`), so a pair taken minutes apart on a later pass is judged differently from one taken two seconds apart;
- **appearance:** a descriptor of each animal's crop. Two are compared on the same out-of-fold pairs: a colour-and-texture descriptor, and features from a pretrained general-purpose vision model (run on the laptop's CPU). Appearance is kept only if it helps (H25): all-black herds may look alike at 2 cm per pixel;
- **size and shape:** box size and aspect after the scale change between the two stills.
These are combined by a logistic model into a probability that the two are the same animal, fitted out of fold on the linked pairs of the verified flights and on simulated flights.

### 5. One global assignment instead of greedy joining (D-9)
The probabilities become costs (log odds) on a graph whose nodes are detections. Grouping them into animals so that the total cost is lowest, with no animal seen twice in one still, is a **correlation clustering** problem: the multi-image form of the assignment the brief asks for (pairwise Hungarian matching is its two-image case). It is solved:
- **exactly** as an integer program (HiGHS, through SciPy) where a connected part of the graph is small enough, which also gives the optimality gap of the heuristic;
- by a **greedy additive edge contraction with local moves** (Kernighan–Lin style) elsewhere; this is the method ported to C++.
It replaces greedy joining only if it beats it (D-30): H23.

### 6. The count and its range
The flight count is the number of animals found, corrected for the detector's expected misses and false detections only if that correction is shown to help out of fold. The 90% range comes from conformal prediction on the relative flight error, calibrated on out-of-fold real flights where there are enough, checked against simulated flights, and reported with its coverage at 68, 90 and 95%. With 20 real flights, real coverage is indicative and said so; simulation gives the precise check (H26).

### 7. C++ core with parity tests
Ported to `core/`: ground projection and covariance, pair evidence (position, movement allowance, size), the clustering heuristic and the count. Each C++ function is checked against its Python reference on recorded flight data and random cases (`tests/test_core_parity.py`), under the sanitizers, in CI. Registration (OpenCV) and the appearance model stay in Python at home base; the C++ core is what the onboard and ROS 2 side reuses (D-14).

### 8. With the Phase 2 detector
When Phase 2 chooses its model, the whole pipeline is re-run with its out-of-fold detections, and S3 is reported for both detectors.

## Hypotheses (written before the first run)
- **H23:** Global assignment (step 5) lowers the mean absolute relative flight error against greedy grouping (D-30), with the same detections and pair evidence, by at least 0.05 on the verified flights, with the paired bootstrap interval excluding zero; and is not worse on simulated validation flights.
- **H24:** The movement allowance lowers the error on flights where the same ground is seen on separate passes more than 2 minutes apart, compared with a fixed distance gate, on the verified and simulated flights.
- **H25:** Appearance improves the same-animal model: out-of-fold area under the ROC curve higher by at least 0.03 than position, movement and size alone.
- **H26:** 90% flight-count ranges contain the truth on 85–95% of simulated validation flights, and real-flight coverage is reported beside it (also H22).
Each rule is written in `configs/phase3.yaml` and frozen before the first run that tests it.

## Order of work
1. Error split (step 1) on all 20 flights; owner reviews the 10 unverified flights in parallel.
2. `configs/phase3.yaml` with H23–H26 and the evaluation design; frozen.
3. Ground positions and covariances; pair evidence; out-of-fold same-animal model.
4. Exact and heuristic clustering; H23, H24, H25.
5. Count correction and conformal ranges; H26.
6. C++ port with parity tests; CI.
7. Re-run with the Phase 2 detector; S3 on validation; Phase 3 exit.

## Alternatives considered
- **Keep greedy grouping (D-30):** simple and already measured; it is the baseline every change must beat.
- **Geotag-only matching:** rejected in D-27 (median 1.5 m error against 2–3 m spacing in herds).
- **Structure from motion / a full orthomosaic, then count once on the mosaic:** strong for static scenes (Shao et al. used 3D reconstruction), but moving animals are smeared or duplicated by mosaicking, and it costs minutes to hours of processing per flight; kept as a comparison on two or three flights if time allows, not the main method.
- **Multi-object tracking (as in video):** stills are seconds apart, not frames; tracking assumptions do not hold.
- **Pairwise Hungarian matching chained across stills:** the chain errors D-9 warns about.
- **A learned matcher (graph neural network) end to end:** needs far more linked flights than exist.
- **The Brazilian geolocated benchmark:** not used: no licence, and the owner decided not to ask (Q10). S3 therefore has no published external comparison; stated as a limitation.

## How success is measured
- S3 on the verified flights, out of fold, with a bootstrap interval, for the Phase 1 and Phase 2 detectors; the same measure on simulated validation flights.
- H23–H26 judged by their frozen rules.
- Parity tests pass in CI under sanitizers.

## Risks
- **Few real flights.** 20 verified flights from three French farms. Mitigation: out-of-fold fitting, paired comparisons, simulation for precise checks; everything reported as indicative where it is.
- **Truth built from label-based candidate links.** The owner checked proposed links and possible missed links, not every possible pairing, so the truth may lean towards the linking it started from. Mitigation: the review also asks about missed links; a sample of animals from each flight is checked in a second, blind pass where the owner counts from the photos without candidate links.
- **Animals that move between passes.** The movement allowance is measured on real pairs, and swept in simulation.
- **Look-alike animals.** Appearance may not separate an all-black herd; position and timing then carry the decision, and this is reported per flight.
- **Detection misses dominate on a new farm.** Linking cannot recover animals the detector never found; the error split (step 1) shows how much is detection, and that part belongs to Phase 2.

## What the owner will be asked to do
1. Agree the plan.
2. Review the 10 unverified flights on the existing link-review page (about 1–2 hours), and a short blind count of a sample of animals per flight (about 30 minutes).

## Review
Self-review as a hostile senior engineer (brief §3.2):
1. *Tuning on the evaluation flights.* The first draft fitted the same-animal model and gates on the 10 verified flights and scored S3 on the same 10. That is tuning on the test of the specification. Changed: out-of-fold fitting by detector fold, and 10 more flights verified.
2. *An easier metric.* S3 is the flight-total error; a method can get totals right by luck (a missed animal cancels a double count). Added: the error split, and pair-level precision and recall of the linking reported beside S3.
3. *Truth bias.* The truth comes from label-based candidate links the owner corrected; a method close to that linking is favoured. Added: the blind count of a sample (Risks).
4. *Mocked parts.* Appearance through "a pretrained model" could become a placeholder. Stated: two named descriptors compared on the same pairs, kept only if H25 holds.
5. *Separate C++ and Python maths.* Covered by parity tests for every ported function; registration and appearance are not ported, so there is no hidden second copy.
6. *Uncertainty.* Every real-flight figure carries a bootstrap interval; real coverage of 20 flights is called indicative, not a pass.
7. *Familiar method.* Correlation clustering is chosen because the problem is multi-image grouping under a one-per-still constraint; plain min-cost flow fits tracking in time order and cannot express that constraint across many stills.
8. *Narrowed scope.* Moving animals and repeated passes are in (H24); calves are counted like any animal; hills affect registration and are reported through registration failures; night and thermal are not in Phase 3 (no data), stated as a limitation.
9. *Best way regardless of effort:* hand-linked identities on many flights from UK and South African farms, so the pair model and the ranges are fitted and checked on target-like data. Not possible in Stage A (no flights); this plan does the next best thing and says so.
10. *Most likely failure on a real farm:* a different flight pattern (more passes, longer gaps) or a dense, uniform herd, where position, movement and appearance all become weak; the count then overestimates. Simulation sweeps both.
11. *How a sceptic would attack the results:* 20 flights, one country, truth from one person. The answer is the out-of-fold design, intervals, the blind check and the simulation, all reported, and no claim that the method generalises beyond what was measured.

## Changes after review
- Out-of-fold fitting for everything learned; 10 more flights to verify (findings 1, 11).
- Pair-level linking precision and recall reported beside S3; error split first (finding 2).
- Blind sample count to check the truth (finding 3).
- Appearance as two named, compared descriptors behind H25 (finding 4).
- Correlation clustering justified against min-cost flow; exact solver used to measure the heuristic's gap (finding 7).

## Progress
- **Step 1, error split** (`scripts/phase3_error_split.py`, `results/phase3/error_split.json`; Phase 1 method, all 20 owner-verified flights, 95% bootstrap intervals over flights). With the labels as detections, the linking alone gives a mean absolute relative flight error of 0.145 (0.093–0.206), and leans towards overcounting (mean signed error +0.069): animals seen twice are too often left as two. With the Phase 1 detector's out-of-fold detections the error is 0.285 (0.170–0.415), so detection adds 0.140 (0.002–0.298). The two parts are about equal, so neither phase can reach the 0.15 target alone. Detection's share is concentrated in a few flights: the farm the detector never trained on (Derval: missed animals, −33% and −52%), one dense single-image herd (Mauron flight 029: −70%), and two flights with heavy overlap where false detections are repeated across many images and not joined (Mauron flights 062 and 065: +108% and +76%). The last kind is partly a linking problem as well, so the same-animal model (step 4) uses detection confidence as evidence too. Linking-only errors are largest on Jalogny flights with few overlapping pairs (overcounts of 11–32%) and on Mauron flight 061 (−53%, animals wrongly joined). On the first 10 flights the figure with the detector (0.264) differs from the Phase 1 report (0.26) only by rounding; on the 10 newly verified flights it is 0.305. Public-data results, not field results.
- **Step 3, pair evidence and the same-animal model** (`herdsight/count/evidence.py`, `herdsight/count/truth.py`, `scripts/phase3_pairs.py`, `results/phase3/pair_model.json`). Candidate pairs within 4 m (registered) or 10 m (geotag) between overlapping stills of the 20 verified flights: 20,435 pairs with labels as detections (7,067 one animal, 12,448 different, 920 unknown inside groups the review left with two boxes from one still), 15,933 with the Phase 1 detector. Each pair is labelled from the owner's review; about a quarter of same-animal pairs (1,868 of 6,896 registered ones) lie beyond the Phase 1 gate, because the review groups whole animals across stills, not only nearest pairs. A logistic model on distance, competition (mutual closest, margin to the next candidate), size difference, detection score and geotag fallback, fitted out of fold, separates the pairs with an area under the ROC curve of 0.966 (0.952–0.982) with distance scaled by the Phase 1 gate and 0.957 (0.943–0.977) with the gate widened by the grazing allowance (labels); 0.962 and 0.952 with the detector. Design change before any grouping result: the first version of the movement scale replaced the Phase 1 gate with a fixed distance in metres, so it changed two things at once; it now widens the same gate by the grazing term (scale² = gate² + 4·D·dt, D fitted out of fold), so H24 compares only the time term. On pairs, the time term does not help; H24 is judged on flight counts in step 4.
- **Step 4, grouping on the real flights** (`herdsight/count/clustering.py`, `scripts/phase3_grouping.py`, `results/phase3/grouping_real.json`; freeze checked first; evidence set and detections for H23 fixed before the run). With labels as detections, the new evidence with global grouping gives a mean absolute relative flight error of 0.091 (0.059–0.126), against 0.125 with greedy joining of the same evidence and 0.145 for the Phase 1 method; pair-level linking precision rises from 0.87 to 0.91 and recall falls from 0.97 to 0.90. With the Phase 1 detector, the errors are 0.270 (global), 0.265 (greedy) and 0.285 (Phase 1): **H23 is not held** (gain −0.005, −0.074 to 0.056). The largest losses are flights where false detections repeat across many overlapping stills (Mauron flight 065: 54 groups against 25 animals, greedy 41): such detections were never labelled as one object (pairs of two false detections are unknown, so they are left out of the model's training), so global grouping, which follows negative evidence more strictly, keeps them apart. The exact solver covers a median 58% of detections per flight (parts of at most 60); on those parts the heuristic falls short of the optimum by up to 6.3 in summed log odds on a flight, so the heuristic is worth improving before the C++ port. H24 cannot be measured on real flights: none has overlapping stills more than 2 minutes apart. Next: the simulated part (H23 non-inferiority, H24), count ranges (H26), then the Phase 2 detector. Public-data results, not field results.
- **H25, appearance** (`scripts/phase3_appearance.py`, `scripts/phase3_h25.py`, `results/phase3/h25.json`): both named descriptors were computed for every detection (3,392 crops for the labels, 3,171 for the detector) and added to the same-animal model out of fold. **Not held**: the colour-and-texture distance lowers the AUC (0.955 against 0.962 without), and DINOv2 raises it by 0.002. Alone, each separates pairs only weakly (AUC 0.71–0.72). Appearance is dropped; position, competition and size carry the decision, as the risks section anticipated for look-alike herds.
- **Step 6, count and ranges on real flights** (`herdsight/count/ranges.py`, `scripts/phase3_ranges.py`, `results/phase3/ranges_real.json`). Counting method: greedy joining with the new pair evidence, since global grouping did not beat it (H23). Ranges are split conformal on the log ratio of true to counted animals, calibrated per fold on the other folds' flights (13–14 flights). The bias correction is dropped: out of fold it raises the error (0.276 against 0.265). Out-of-fold coverage: 0.75 at 68% (mean width 0.83 of the herd), 0.95 at 90% (width 2.71), and no finite 95% range with this few calibration flights. The 90% ranges are honest but too wide to be useful: with 13–14 flights the 90% quantile is set by the two worst flights (−70% and +76%, detection failures). They narrow only as the detector improves; Phase 2's model is the lever. Real-flight coverage of 20 flights is indicative; H26 is judged on simulated flights.
- **Simulated validation flights** (`herdsight/sim/phase3.py`, `scripts/phase3_sim.py`, `results/phase3/sim_validation.json`; 200 flights on the validation seed, position-only pair models fitted on the 20 real flights). Mean absolute relative error: Phase 1 0.347, greedy with the new evidence 0.321, global 0.334; all overcount strongly (+0.31 to +0.34), far more than on real flights, as H20 and H21 found (false detections on fixed objects and duplicates in the simulation's detection model). **H23** simulated part: global is worse by 0.013 (0.008–0.018), beyond the 0.01 margin. **H24 not held**: +0.002 (−0.001 to 0.005) on the 136 flights with overlapping stills more than 2 minutes apart. **H26 not held**: real-calibrated 90% ranges cover 97.5% of simulated flights, above the 85–95% band (too wide, not too narrow). **H22 (Phase 5) held**: simulated 90% coverage 97.5% against 95% real. Simulation results, not field results.
- **Where Phase 3 stands.** The method kept is greedy joining with the new pair evidence (distance in Phase 1 gate units, competition, size and detection score; no appearance, no movement allowance, no global grouping, no bias correction). It lowers the linking-only error from 0.145 to 0.125 and, with the Phase 1 detector, the flight error from 0.285 to 0.265 (not a pre-registered comparison; reported as measured). The S3 target of 0.15 is not reached with the Phase 1 detector; the error split shows the detector is now the main source. Next: the C++ port of the kept method with parity tests, and the re-run with the Phase 2 detector.
- **Step 7, C++ port** (`core/include/herdsight/count.hpp`, `core/src/count.cpp`, `core/tests/test_count.cpp`, bindings in `core/python/bindings.cpp`): the kept method (pair features with the competition terms, the logistic score and greedy joining with the one-per-still rule) is in the C++ core. Six GoogleTest cases pass under the address and undefined-behaviour sanitizers with warnings as errors, and `tests/test_core_parity.py` checks the C++ against the Python reference on random cases (features to 1e-12, log odds to 1e-9, identical groupings) and on the recorded pair tables of the 20 verified flights (identical counts on every flight). Ground projection with covariance and the global grouping heuristic are not ported: the first is not used by the kept method, and global grouping was not adopted (H23).
