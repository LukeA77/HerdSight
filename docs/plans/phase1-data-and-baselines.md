# Phase 1: data audit and baselines

## Goal

Know exactly what data there is, how it is split, how results are scored, and how well simple methods do. Then set targets for S1, S2, S3 and S7 that are ambitious but grounded, and freeze them.

Exit criteria (brief §12): a frozen specification file and baseline results in `results/`.

## Approach

1. **Audit.** Availability, licence, format and size of every candidate dataset, in `docs/data.md`. Licence policy: D-21.
2. **Inspect ICAERUS v2** (the main source):
   - read every image's EXIF: GPS, relative altitude, gimbal angles, focal length, sensor model, time;
   - check label quality on a sample of images;
   - measure per-flight overlap between neighbouring images from the GPS track and camera footprint.

   Outputs: the reference camera for flight planning (D-6), and whether ICAERUS can serve as the S3 benchmark.
3. **Splits** (brief §8.2), written to `results/splits.json` before any model is run:
   - **Test:** one whole farm held out. The farm is chosen by a rule written down before any results exist: the farm whose box count is closest to 20% of the total, ties broken alphabetically. Its images are not opened by any model until Phase 7.
   - **Calibration** (conformal intervals) and **validation**: whole flights from the remaining farms, never single images.
   - **Shao et al.** (Japan, non-commercial): a second unseen-source test for evaluation only. It measures how far results carry to a different country, breed and camera.
4. **Loaders** (D-4). One record type, `SurveyImage` (image path, camera model, GPS, altitude, attitude, time, optional labels). It is filled from:
   - ICAERUS (PASCAL VOC labels);
   - Shao et al.;
   - **any folder of geotagged drone stills with no labels**, which is what a real farm flight produces.
5. **Evaluation** (brief §9), with tests on hand-made cases:
   - **centre matching:** Hungarian one-to-one matching of predicted and true centres, scored with an Object Keypoint Similarity (OKS)-style score `s = exp(-d² / (2 (κ·σ)²))`, and AP over thresholds 0.50:0.05:0.95;
   - **upright-box AP50 and AP50:95** through pycocotools, so results compare with published work;
   - **count error:** mean absolute error per image, and for S3 per flight;
   - **interval coverage** at 68%, 90% and 95%;
   - **bootstrap intervals** over flights.

   κ is set from measured label noise. The owner re-draws boxes on 100 randomly chosen training cattle, blind to the originals, and the spread of the centre differences sets κ. Until then κ stays an unresolved placeholder that the code refuses to use.
6. **Run-once guard.** Test-split scripts refuse to run if the frozen settings changed (`scripts/freeze.py --check`) or if a test result already exists.
7. **Hypotheses** written in `docs/hypotheses.md` before any baseline runs.
8. **Baselines, on validation only:**
   - **S1/S2 (a), zero-shot:** an open-vocabulary detector with no cattle training (OWLv2, Apache-2.0), on full-resolution overlapping tiles. It shows what is possible without any training.
   - **S1/S2 (b), quick fine-tune:** an upright-box detector with a licence that allows a product (D-FINE, Apache-2.0; tried in drone-ranger), trained briefly on Kaggle on full-resolution tiles from the training farms. Default settings; no tuning.
   - **S2 intervals:** split conformal intervals on per-image counts from baseline (b), fitted on calibration and checked on validation.
   - **S3:** only if step 2 finds overlapping, geotagged flights. Ground truth comes from hand-linked animal identities on a few flights (brief §8.3). Two baselines: the naive sum of per-image counts, and merging detections within a fixed ground distance.
   - **S7:** herd recall of a small detector on images downscaled to an onboard budget. Latency is reported as **not measured** without a Raspberry Pi 5. A laptop CPU figure may be given, labelled as such, never as Pi speed (D-10).
9. **Targets:** written into `configs/specs.yaml` from the baselines and published results, then frozen with its hash (D-2).

## Alternatives considered

- **Split by image:** rejected (brief §8.2). Images from one flight are near-duplicates.
- **Hold out a random set of flights instead of a farm:** this measures less of what matters, since a new farm differs in breed mix, pasture and light. A farm hold-out is the closer stand-in for "a farm the system has never seen".
- **Baselines on the test farm:** rejected; see review.
- **Ultralytics YOLO-OBB as a baseline:** strong and easy, but AGPL-3.0. Its use is limited to a separate, clearly labelled comparison in Phase 2, if at all, so no product code depends on it (as drone-ranger did, D-20 context).

## How success is measured

- Every dataset decision traceable in `docs/data.md`, with checksums.
- `results/splits.json` committed before any baseline result exists; git history shows the order.
- Evaluation code tested on hand-made cases with known answers, and the upright-box AP checked against pycocotools on the same input.
- Baseline results in `results/baselines/`, each with environment, seed, config hash, and bootstrap intervals.
- `configs/specs.yaml` frozen, with its hash in `results/frozen/`.

## Risks

- **Domain shift.** All usable cattle data is French: European breeds, green pasture, one camera family. South African veld, Nguni and Brahman colours, and harsh midday shadows are not represented. Shao et al. (Japan) measures part of this gap, not all of it.
- **Small data.** 1,385 images and 4,941 boxes across four farm groups, so intervals will be wide.
- **The Brazilian benchmark may be unavailable**, or available only without permission to train.
- **Free GPU limits** (Kaggle weekly quota) slow the fine-tuned baseline.
- **Many empty images** in some ICAERUS flights. Precision on empty images matters for false alarms, so they are kept, not dropped.

## Review

**Red flags checked:**

- *Shrinking images:* both detection baselines run on full-resolution tiles. S7 downscales on purpose, because it models the onboard budget, and says so.
- *Easier metric:* the main score is centre matching with AP over a range of thresholds (brief §9.1), not a single lenient threshold.
- *Test data related to training data:* the test set is a whole farm; calibration and validation are whole flights. One weakness remains: all ICAERUS farms share a drone family, region and annotation team, so the farm hold-out is easier than a truly new farm. Shao et al. is the second, harder test.
- *Mocking:* none planned. S3 is explicitly conditional on the data, not stubbed.
- *Narrowing scope:* calves are not labelled separately in ICAERUS. The resolution study for calves (Phase 2) needs calf sizes, which this phase records as box-size distributions.
- *Single numbers:* every baseline reports bootstrap intervals over flights.
- *Assuming without measuring:* image overlap, GPS quality and label noise are each measured, not assumed.
- *Familiar rather than best:* D-FINE is chosen for the baseline partly because it is familiar. That is acceptable for a baseline, which only needs to be honest. The architecture choice proper is Phase 2's comparison of at least two families (D-8).
- *Tuning on the test set:* **the first draft ran baselines on test data; corrected below.**

**1. What is the best way, ignoring effort?** Own labelled survey flights from UK and South African farms, with kraal counts as ground truth. That is impossible in Stage A (no farm access). The plan instead designs loaders so that such data drops in later, and states the domain gap plainly.

**2. What will most likely make this fail on a real farm?** Appearance shift: cattle colours and backgrounds unlike France. Dark cattle in deep shade, and pale cattle on pale dry grass, are the likely failures. The baseline reports per-flight results, so the hardest conditions in the data are visible rather than averaged away.

**3. How would a sceptical reviewer attack it?**

- "Targets were set by looking at the test set": no; targets come from validation baselines and published numbers, and the test farm is untouched until Phase 7.
- "κ was chosen to flatter the model": κ comes from blind re-annotation by a person, before any model runs.
- "A held-out farm from the same project is not a new farm": agreed, which is why Shao et al. is also tested and why the limitation is stated.

## Changes after review

- Baselines run on **validation**, not test. The first draft said "baselines for S1–S3" without naming the split, which would have spent the test set before Phase 7.
- The test-farm choice is fixed by a written rule based on box counts, decided before anything is run.
- κ is set by blind re-annotation by the owner, not chosen by the developer.
- Empty images are kept in evaluation, so false positives on empty pasture are counted.
- Box-size distributions per flight are recorded, as input to the calf resolution study.

## Changes during the build

These were found while building, before any model was run, and are recorded here so the order is clear.

1. **Splits by farm and date, not by flight (D-25).** The first splits file put flights flown minutes apart over the same herd into different splits. Replaced before any model ran.
2. **Grouped cross-validation instead of one train/calibration/validation split (D-26).** With date grouping, a single split left validation as one farm on one day. Three folds over the 10 farm-date groups score every non-test image once with a model that never saw its farm and day. The operating threshold and conformal intervals for each fold are fitted on the other folds. The hypotheses were reworded to match before any model ran; their predictions are unchanged.
3. **YOLO labels, not VOC.** The VOC labels miss 306 images; the YOLO labels cover all of them and agree with VOC within 1 pixel where both exist (docs/data.md).
4. **Zero-shot sample 10% of non-test images** (D-24), because cross-validation needs every non-test image scored and the CPU runs at about 4 minutes per image.
5. **No calf labels exist** in ICAERUS (the class is defined but unused). The calf resolution study (D-6) needs another source of calf sizes; this is carried to Phase 2.
6. **S7 scored on herds, not single animals (D-28).** Herd recall is reported two ways: *found* (at least one detection inside one of the herd's animals, D-28's definition) and *grouped* (the detections themselves form a herd that touches it, which is what herd mode would react to). False triggers are predicted herds that touch no true herd. The onboard model's threshold per fold follows the same other-folds rule as the main scorer.
7. **Laptop dropouts.** The external drive dropped out once and the laptop went into standby once during long runs. Both stopped jobs part-way; every long job resumes from its saved lines, and image reads retry. No result depends on an interrupted run's partial output.

## Verification

*(Filled in as each step completes.)*

- **Audit:** `docs/data.md`. ICAERUS downloaded and checked against its MD5; inventory in `results/data/icaerus_inventory.json`.
- **Splits:** `results/splits.json`, frozen with its hash at 2026-11-18T17:08Z, before any model was run (`results/frozen/phase1_splits.json`).
- **Evaluation code:** tested on hand-made cases (centre matching, COCO AP, counts, conformal intervals, tiling ownership, YOLO/VOC loading, splits). End to end, the scoring script was run on "oracle" predictions equal to the labels. It returned count MAE 0, AP50 1.0, recall 1.0, precision 1.0 and interval coverage 1.0, as it must.
- **Zero-shot detector check** on one image (9 cows): coordinates correct (all 9 matched at IoU 0.5 at score ≥ 0.2).
- **S3 scorer:** on labels used as detections, the registration method reproduced the label-based candidate count exactly on all 20 flights, so the scorer and the linking script agree. Two of the 20 flights are a single image with animals; they cannot overcount and are left out of H6's test.
- **Early sign on H6, not a result:** against the *unverified* candidate counts, the naive sum overcounts by 14% on two Jalogny flights (DJI_202309151045_055 and _056), below H6's predicted 20%. Neither flight is in the owner's link review, so this is not tested yet. H6 stays as written; it is scored only against verified counts.
- **Herd scorer (S7):** tested on hand-made cases; end to end on oracle predictions it returned herd recall 1.0 (found and grouped) and no false triggers over the 339 herds in the cross-validation images.
- **Centre-matching constant κ:** from the owner's blind re-annotation of 100 random training-fold cattle (94 answered, 6 skipped as unclear): κ = 0.076, 95% interval 0.063–0.090 (`results/data/kappa_fit.json`). Set in `configs/evaluation.yaml` before any baseline was scored with it.
- **S3 ground truth, round 1:** the owner's link review of 3 flights (234 checks). The candidate links had missed many true links: the verified counts are 135, 68 and 35 animals, against 163, 81 and 37 candidates (6–21% too high). A few joins put two boxes from one image together. Each such group is counted as at least the number of animals its most-repeated image shows (`results/data/s3_truth.json`). Because the unverified candidates are this far off, S3 and H6 are scored only on verified flights. A second review round adds one flight from each farm-date group not yet covered (7 flights, 445 checks), by a rule fixed before any detector was scored on S3.
- **S3 linking with perfect detections** (labels used as detections, verified flights only): the registration method's per-flight count is 15% from the verified count on average, geotag-only matching 38%, and the naive sum 59% to 950% too high. This is the error from linking alone, before any detector error; the detector-based S3 baseline adds to it.
- **S3 ground truth, round 2:** the owner reviewed the 7 further flights (all 445 checks answered). Round 1's three counts are unchanged, and 10 flights are now verified, covering every farm-date group in cross-validation. On these flights the candidate links again overcounted (for example 158 candidates against 120 verified animals on Jalogny DJI_202308031317_042); on one Derval flight they undercounted (21 against 27), because the review split off pictures that had been linked to the wrong animal.
- **S3 linking with perfect detections, all 10 verified flights:** registration 15% from the verified count on average, geotag-only 21%, and the naive sum 27% to 1,060% too high (every flight above H6's 20%). This is not H6's test, which uses the detector's counts.
- **S1 scoring completed before any baseline was scored:** the scorer now also reports centre-matched precision and recall at the operating point (similarity ≥ 0.5, `configs/evaluation.yaml`), which is what S1 asks for. The IoU-based figures stay as the upright-box comparison.
- **How targets are set (D-31):** a fixed rule was proposed, then replaced by the owner's decision to set each target after reviewing the scored baselines and their failure cases. `scripts/baseline_summary.py` gathers the evidence for each target; the targets are frozen before any Phase 2 work.
- **Hypothesis rules fixed in code before the fine-tuned baseline was scored:** `scripts/judge_hypotheses.py` turns each written prediction (H1–H6) into a check and writes `results/baselines/hypotheses.json`. H1 asks for non-overlapping intervals on centre AP, so the scorer now also gives centre AP a 95% interval (resampling farm-date groups, as for the other intervals). The interval is drawn after all the others, so every other number is unchanged; this was confirmed by rescoring zero-shot (identical apart from the new interval: centre AP 0.25, 0.15–0.45).
- **Kaggle training:** each full-resolution run took about 3.2–3.8 hours on a T4 GPU, so the six runs were split over three sessions. Two problems were fixed on the way: the first ONNX export ran out of memory (fixed by exporting with a 2-image dummy input and gradients off, which gives a byte-identical file; the first checkpoint was exported locally with `scripts/export_dfine_onnx.py`), and a new release of `faster-coco-eval` (1.8.1) broke D-FINE's evaluation, so the earlier working version (1.8.0) is pinned.
- **Zero-shot baseline (S1/S2 (a))**, out-of-fold on the 133-image sample (415 cattle), from `results/baselines/zeroshot_scores.json`:
  - count MAE 0.95 animals per image (95% interval 0.56–1.73); exact count on 79% of images; 90% interval coverage 0.92;
  - centre-matched recall 0.48 (0.36–0.70), precision 0.52 (0.42–0.67); centre AP 0.25;
  - upright-box AP50 0.74, with recall 0.67 and precision 0.72 at IoU 0.5.

  Centre matching with κ = 0.076 is stricter than IoU 0.5 for these small animals, which is why its recall is lower.
- **Fine-tuned D-FINE baselines** (all six Kaggle runs completed 72 of 72 epochs; every model checked against its recorded ONNX hash). Out-of-fold over all 1,323 cross-validation images (3,392 cattle, 10 farm-date groups), from `results/baselines/dfine_scores.json`:
  - S1: centre-matched recall 0.72 (0.63–0.82), precision 0.70 (0.60–0.81); centre AP 0.54 (0.42–0.69); upright-box AP50 0.87 (0.79–0.93);
  - S2: count MAE 0.66 animals per image (0.36–1.04); exact count on 74% of images; 90% interval coverage 0.93 (0.87–0.98);
  - S3 (`dfine_s3.json`, 10 verified flights): mean absolute error of the flight total 26% with registration, 35% with geotags only, 270% for the naive sum;
  - S7 onboard (`dfine_onboard_herds.json`, 339 herds): herds found 0.94 (0.88–0.98), found as a group 0.75 (0.60–0.87); 0.033 false herd triggers per image. Latency not measured (D-29).
- **Failure review** (`scripts/failure_review.py`, `results/baselines/dfine_failures.json`): the detector misses 67% of cattle on Derval, the only farm scored by a model that saw no image of that farm, against at most 19% on any other group; 443 of 483 missed animals were detected but scored below the cut-off; misses are most common for small, touching and edge-cut animals; 68% of false detections come from two farm-date groups (farmyard objects, bales, plastic, people, pale stones). Found animals' box centres are off by 0.079 of the short side (root mean square), close to the hand-labelling spread κ, so centre-matched recall is capped near 0.85 with these labels. Two images with empty label files show herds (D-32).
- **Hypotheses** (`results/baselines/hypotheses.json`): H3 held; H1, H2, H4, H5 and H6 not held. See `docs/hypotheses.md` for each outcome.
- **Targets:** set by the owner after this review (D-31), with reasons, in `configs/specs.yaml`, and frozen with the baseline evidence (`results/frozen/phase1_specs.json`) before any Phase 2 work.
