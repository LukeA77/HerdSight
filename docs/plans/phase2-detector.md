# Phase 2 plan: detector

Status: revised after independent review and agreed with the owner in its lean form (2026-11-21; see "Owner decisions").
Inputs: Phase 1 baselines and failure review (`results/baselines/dfine_scores.json`, `dfine_failures.json`), frozen targets (`configs/specs.yaml`, `results/frozen/phase1_specs.json`).

## Goal
A home-base cattle detector that meets the frozen S1 and S2 targets on validation (grouped three-fold cross-validation, D-26), works as well as possible on farms it has never seen, outputs oriented boxes (D-8), has a density cross-check, and gives calibrated count intervals. Also: the pixels per animal needed for calves, which sets flying height (D-6).

Exit (brief §12): S1 and S2 measured on validation, and a written comparison of candidates.

## What Phase 1 showed
| Finding | Evidence |
|---|---|
| The one farm scored by a model that never saw it (Derval) loses 67% of its cattle; half score below 0.3 | `dfine_failures.json` |
| Derval was flown with the M3T camera (4,000 px wide). At 60 m its animals are about 102 px long (median of each image's largest box), against about 150–160 px on the other farms' 60 m flights (M3E, 5,280 px). The only M3T images in training are Mauron at 100 m (about 66 px), which itself misses 15–19% | `results/data/icaerus_images.csv` |
| 443 of 483 misses are detected but scored below the cut-off | `dfine_failures.json` |
| Small (under 28 px short side), touching and edge-cut animals are missed most | `dfine_failures.json` |
| False alarms are farmyard and pale objects, 68% from two farm-date groups | `dfine_failures.json` |
| 90% intervals cover 78% on images with 20+ animals (37 images) | `dfine_scores.json` |
| 1,019 of 1,323 validation images are empty; the test farm has none and up to 90 animals per image | `docs/data.md` |
| The baseline trained 72 epochs, not the authors' 220, to fit one free session; D-22 promised Phase 2 would train properly | `docs/decisions.md` D-22 |

## Approach

The work is split into a **CPU stage** (no GPU time, done first) and a **GPU stage** with a fixed order and a run ledger. Every run is all three folds unless stated. Nothing in the GPU stage starts before the CPU stage's pre-registration (C3) is written.

### CPU stage (laptop, no Kaggle time)

**C1. Root-cause the new-farm failure.** Split Derval's misses by flight, altitude, camera, animal size in pixels and in ground units, shade and posture; compare score histograms with Mauron's M3T flights; look at the 15 worst images. Output: `results/baselines/derval_rootcause.json` and a short note. This decides which levers in G2 are tried first.

**C2. Hypotheses H7 onwards** written in `docs/hypotheses.md` before any GPU run, for example:
- H7: the combined appearance and crowding recipe raises Derval recall by at least 0.15 and costs no more than 0.02 pooled recall;
- H8: nested hard-negative mining at least halves false detections per empty image;
- H9: training with the full schedule raises centre-matched recall by at least 0.03 over 72 epochs;
- H10: oriented boxes raise recall on touching animals by at least 0.05;
- H11: band-wise intervals reach at least 0.85 coverage on 20+ images with mean width no more than 1.5× the pooled interval's;
- H12: the larger model (M) beats S on centre AP by more than the seed spread.
Each prediction is judged by a rule fixed in code before the run, as in Phase 1.

**C3. Pre-registration** (`configs/phase2.yaml`, frozen with `scripts/freeze.py --phase phase2_prereg` before the first GPU run):
- the candidate list and the order of the GPU stage;
- the selection rule: a candidate replaces the current best only if its paired bootstrap difference over farm-date groups (same images, both models) is better on centre AP **and** count MAE with intervals excluding zero, the difference exceeds the seed spread (G1), and it does **not** lose on worst-group recall or on leave-one-farm-out recall (G6);
- count-band edges (0, 1–4, 5–19, 20+), the conformity score (residual divided by √(prediction + 1)), and a width cap for intervals;
- reporting: every candidate's numbers are reported, not only the winner's; validation results for the chosen model are labelled as selection-biased.

**C4. Training-data hygiene.**
- The two D-32 images with unlabelled herds, and any further empty-labelled training image that the owner confirms contains cattle, are **excluded from training** (scoring is unchanged, D-32).
- To find them without bias in scoring: every empty-labelled image where any Phase 1 model fires with confidence 0.9 or more (71 images) goes into a quick review page; the owner marks "cattle" or "no cattle". Used only to exclude training images.

**C5. ICAERUS raw images** (CC BY 4.0, about 900 unlabelled): match each to v2 by file name, timestamp and GPS. Rules: anything within 2 km of any Other_farms image or taken on a test-farm date is excluded from everything (training, statistics, self-training); anything matching a cross-validation group goes into that group's fold; only images from new places or days are kept, and they are screened by the owner before use as negatives. If none qualify, this step is dropped and said so.

**C6. Oriented-box labels and their centres.** Run `herdsight.labels.sam_obb` (SAM ViT-B) on every training box.
- Score the SAM boxes as if they were predictions against the published upright labels (centre-matched recall, centre AP). If centre-matched recall is below 0.98, the oriented box's centre is fixed to the upright box's centre and SAM supplies only angle and size.
- Flagged labels (`no_mask`, `escapes_box`, `low_fill`) fall back to an angle-0 box of the upright size, never dropped (dropping would leave unlabelled animals among the touching ones).
- 100 labels, stratified by fold and flag, are checked by eye; acceptance rate set beforehand at 90%. If it fails, OBB training is paused and the cause fixed.
- Near-square boxes: the angle loss handles 180° periodicity; tests cover it.

**C7. Hand-verified rotated subset (§9.3).** Chosen now by a label-only rule (most touching or overlapping labelled boxes): about 150 animals from the cross-validation folds **and** about 150 from the test farm (scored once, in Phase 7). The owner draws from scratch, never over SAM boxes; 30 boxes are redrawn a week later to measure the owner's own spread.

**C8. Move scoring inference to Kaggle**, straight after training, so the laptop is not tied up for about 9 hours per candidate.

**C9. New code and tests:** rotated IoU (checked against shapely), the enclosing upright box of an oriented box (so oriented candidates report upright AP50 and AP50:95, §9.2), centre ownership for rotated boxes, band-wise conformal intervals, paired bootstrap differences. A C++/Python parity test for anything that enters the C++ core.

### GPU stage (Kaggle, fixed order)

**G0. Does 72 epochs converge?** One fold of the reference recipe at the authors' full 220 epochs, resumable across sessions, compared with its 72-epoch run on a **training-side** hold-out (one farm-date group held out from the training folds; the monitor tiles of Phase 1 come from the validation fold and cannot be used). If 220 is clearly better, all finalists train at 220 (D-22). If not, 72 is kept with that evidence.

**G1. Noise floor, only where a decision is close.** Training randomness is measured by repeat runs (seeds 1 and 2, all three folds) only when a comparison is close: when the paired difference on centre AP or count MAE is under 0.03, or its interval crosses zero. A clear win is accepted without repeats. Any repeats run for the reference recipe set the noise floor for later comparisons.

**G2. One combined recipe first.** The changes with the strongest evidence (C1) are combined into one recipe, written down before the run and not tuned per fold:
- appearance: brightness and contrast shifted towards dark animals and shade, colour jitter, random greyscale, synthetic shadow, blur, 90° rotations (nadir images have no "up");
- crowding: tiles with touching animals oversampled, and labelled animals pasted into groups on other tiles (copy-paste augmentation, training folds only);
- mild scale jitter;
- the nested hard negatives of G3;
- the known unlabelled-cattle images excluded (C4).
It is compared with the reference recipe. Only if it fails to beat the reference are its parts tested one at a time. GSD normalisation (resampling every image to a common ground sample distance) is held in reserve: C1 found that image scale does not explain the Derval failure.

**G3. Nested hard-negative mining (proposed D-34).** For fold k with training folds a and b: train on a, mine false detections on b; train on b, mine on a. The mining models train at half the schedule (they only find confusing objects and are never scored). Crops are screened by the owner (no animal), and added as background at a ratio fixed beforehand (one hard negative per two positive tiles). No fold's labels ever influence the negatives used for that fold.

**G4. Model size (D-8: "architecture and size").** D-FINE M on the best recipe so far; L if the budget allows and M beats S by more than the seed spread.

**G5. Small animals: scale.** Compare GSD normalisation at a finer target GSD with a fixed tile enlargement. Tile overlap is kept at or above the largest animal length in source pixels after scaling, so centre ownership stays exact (D-23). Edge-cut animals are handled separately: partly visible animals stay labelled, and image borders are padded.

**G6. Leave-one-farm-out (proposed D-33).** For the reference recipe and the final model, each of Mauron, Derval and Jalogny is predicted by a model trained only on the other two. Reported beside the official cross-validation, with worst-group recall and count error. A candidate may not be chosen if it loses here (C3).

**G7. Second architecture family and oriented boxes.** Oriented boxes are an **output requirement** (herd-mode framing, later identification), not assumed to improve accuracy: D-FINE has no non-maximum suppression and tiles merge by centre ownership (D-23), so D-8's original reason does not apply as written. Therefore:
- Candidates: an oriented detector from a maintained, permissively licensed codebase (RTMDet-R via mmrotate 1.x / mmdet 3, Apache-2.0; fallback detectron2 rotated Faster R-CNN, Apache-2.0); a DETR-style oriented detector if one with a permissive licence and working code is found before building a custom angle head; and, as the last option, D-FINE with an angle head.
- Each gets the **authors' own schedule** for that model and the same data recipe (GSD, augmentation, hard negatives), no extra tuning.
- Decision rule, written now: if the best oriented model is not worse than the best upright model on S1/S2 beyond the seed spread, it ships. If it is worse, the upright detector does the counting and an angle is predicted for each counted animal, and D-8 is superseded with a new decision.
- Two families are compared (D-8): the oriented family against upright D-FINE. The first oriented family tried is RTMDet-R; a second oriented family is tried only if the first cannot be installed or clearly underperforms, with the reason logged.

**G8. Density cross-check.** A point-supervised density head (separate small head, detector weights frozen, so the detector is unchanged). Its count is reported as a counting candidate in its own right, especially on dense images. Its disagreement threshold is fitted on the other folds to maximise 20+ coverage at the smallest mean width. Ablation: count bands alone against bands plus the dense flag.

**G9. Calibration and intervals.** Band-wise conformal intervals with the pre-set edges and score (C3), fitted cross-conformally (each fold's thresholds and quantiles come from models that did not train on that fold, using the G6/G3 nested models where they exist; otherwise the small leakage of the Phase 1 method is stated). Coverage at 68/90/95% overall and per band with intervals, mean width beside every coverage, and coverage under leave-one-farm-out, where the exchangeability assumption is weakest.

**G10. Onboard herd detector (S7).** The final recipe on D-FINE N, rescored on S7.

**G11. Resolution study for calves (D-6).** The G2 model, whose wide scale jitter covers the tested range (no separate training run). Validation images are downsampled with area resampling plus realistic blur, noise and JPEG compression. Pre-set criterion: the smallest animal length at which recall stays within 0.05 of native resolution (with its interval). Calf body length by age from cited sources converts this into a ground sample distance and a flying height. Stated as a lower bound.

### Run ledger and compute
Lean plan agreed with the owner (2026-11-21). Hours are Kaggle T4 equivalents (72-epoch D-FINE S fold run about 3.5 h; 220 epochs about 10.7 h); on the rented RTX 4090 (D-35) they are expected to take a third to a fifth as long, measured on the first session.

| Priority | Item | T4 hours |
|---|---|---|
| 1 | G0 convergence check | 11 |
| 1 | G2 combined recipe, incl. G3 nested mining at half schedule | 19 |
| 1 | G4 model size M | 19 |
| 1 | G6 leave-one-farm-out (reference + final) | 23 |
| 2 | G7 one oriented family + debugging | 19 |
| 2 | G8 density head | 6 |
| 2 | G10 onboard refresh | 4 |
| 2 | G11 resolution study (inference only) | 3 |
| 2 | C6 SAM labels | 1 |
| — | 15% for failed or repeated runs | 16 |
| | **Total if 72 epochs is enough** | **about 130** |
| — | Final model at the full schedule if G0 requires it | +35 to +55 |
| reserve | G1 repeat runs, only for close decisions (about 21 each) | as needed |
| reserve | G2 parts one at a time, only if the combined recipe fails | about 26 |
| reserve | G5 scale, or GSD normalisation | about 26 |

The ledger (`results/phase2/run_ledger.csv`, one row per run, written by the run scripts) records GPU type, hours and cost per run. If money or time runs short, reserve items go first, then priority 2, and every cut is written in the Verification section and the limitations; nothing is dropped silently.

## Out of scope for Phase 2 (stated, not hidden)
- **Thermal and night** (D-5): no usable thermal cattle data at survey height (`docs/data.md`). Limitation.
- **Southern African and UK backgrounds and breeds**: no licensed data. Hard negatives are French farmyard and pasture objects only. Background-only aerial imagery with a permissive licence (bush, termite mounds, red soil) is searched for and added as negatives if found. Limitation otherwise.
- **Calves**: inferred from adult cattle (G11), not measured.

## Alternatives considered
- *Ultralytics YOLO-OBB:* AGPL-3.0; rejected unless a licence is bought.
- *Stay upright only:* kept as the counting fallback by G7's rule.
- *Lower the cut-off:* the cut-off already minimises count error; calibration (G9) is the right tool.
- *Fix the unlabelled images in scoring:* rejected in D-32; they are excluded from training only (C4).
- *Pseudo-labelling unlabelled images:* only if G2/G6 leave a large new-farm gap; generated labels can reinforce mistakes.

## How success is measured
- S1 and S2 on the frozen folds, pooled out-of-fold, with 95% intervals over farm-date groups, against `configs/specs.yaml`; plus **paired** differences against the baseline with intervals. A point estimate inside the baseline's interval is not presented as evidence of improvement.
- Count MAE also reported on non-empty images and per band, because the test farm has no empty images.
- Leave-one-farm-out and worst-group figures for the reference and finalists.
- Rotated IoU only on the hand-verified subsets.
- Seed spread for the reference and final model.
- A written comparison table: accuracy first, then measured laptop CPU and Kaggle GPU speed, each labelled with its hardware.
- The test farm and Shao et al. stay closed until Phase 7.

## Risks
- **The new-farm gap may not close.** Reported as such; it is the main risk to the project.
- **Compute:** rented RTX 4090 within a $30 prepaid cap (D-35). If the measured speed is below the estimate, the owner is asked before spending more. Kaggle stays available for small jobs.
- **Selection bias:** about 6–8 candidates compared on 10 farm-date groups from 3 farms, with recipe steps chosen after the failure review. Mitigated by pre-registration, paired tests, seeds and reporting every candidate; the Phase 7 test farm gives the unbiased number.
- **The test farm differs:** no empty images, up to 90 animals, one camera. Interval coverage above 53 animals per image is an extrapolation; stated in `docs/limitations.md`.
- **Install problems** for oriented-detector codebases on Kaggle; fallbacks named in G7.
- **SAM masks** may merge animals or include shadows; C6 measures and limits the effect.

## What the owner will be asked to do
1. Start RunPod sessions (step-by-step each time) and download the outputs from Kaggle to the storage drive.
2. Review 71 "empty" images for cattle (C4, about 20 minutes).
3. Confirm hard-negative crops (G3, about 30–60 minutes).
4. Check 100 SAM oriented boxes (C6, about 30 minutes).
5. Draw rotated boxes from scratch: about 150 animals from the folds and 150 from the test farm, plus 30 repeats a week later (C7, about 5–6 hours in total).

## Review
An independent reviewer (no part in writing the plan) reviewed the first draft as a hostile senior engineer against brief §3.2 and §14. Main findings, all checked against the files before acting:
1. The first draft kept the 72-epoch schedule that D-22 said Phase 2 would replace (confirmed in `docs/decisions.md`), and fixed the model size at S without testing (brief D8 says architecture and size). Red flag: shrinking to save time.
2. It assumed the Derval failure was colour and shade without measuring it. The data shows a camera and image-scale difference as a strong alternative explanation (confirmed in `icaerus_images.csv`), and Phase 1 already used colour augmentation (confirmed in the training script). Red flag: untested assumption.
3. Hard-negative mining on a fold's own training images is either nearly empty or biased towards unlabelled cattle; out-of-fold mining leaks fold labels. Training still included the known unlabelled-cattle images.
4. No run-to-run variance: every candidate was one seed, and differences were judged by overlapping intervals rather than paired tests.
5. SAM-derived oriented labels can move box centres away from the labelled centres that S1 scores, which could lower S1. And D-8's reason for oriented boxes (suppression deleting animals) does not apply to D-FINE, so their accuracy benefit was assumed.
6. The comparison budget was undefined, the fallback could leave one family (breaking D-8), and adding an angle head to D-FINE was chosen partly because the pipeline is familiar.
7. The compute estimate (4–6 weeks) was too low by about half and had no cut list, so candidates could be dropped silently.
8. Count error is dominated by empty images the test farm does not have; the 0.45 target sits inside the baseline's interval; intervals could meet coverage by widening; the 20+ band has 37 images.
9. The density step had no fitting objective; the resolution study measured robustness of a native-resolution model, not pixels needed; the rotated subset was not from the test farm and not blinded.
10. Pretraining licence policy was inconsistent (DOTA excluded, ImageNet-pretrained backbones not considered); the raw-image check had no exclusion rule; scope was narrowed without saying so (thermal, non-French backgrounds and breeds); hypotheses for Phase 2 were missing; the claim that nothing can beat the centre-placement ceiling was untested (κ includes two people's noise).

Answers to the brief's questions:
1. *Best way regardless of effort:* labelled flights from several UK and South African farms at the planned GSD, with calves and local breeds; leave-one-farm-out as the main measure; large models on full schedules; several seeds. This plan cannot get target-farm data (no licensed source exists, and no farm flights in Stage A), so it does the next best things: measures new-farm transfer directly (G6), attacks the measured cause (C1, G2), trains properly (G0, G4) and reports variance (G1). The lack of target-farm data is stated as the main limitation.
2. *Most likely failure on a real farm:* a different camera, height, breed or background makes scores fall below a threshold fitted in France, so animals are silently undercounted and intervals stay narrow while wrong. G2, G6 and G9's leave-one-farm-out coverage measure exactly this, and Stage B's go/no-go gates must test it on farm data.
3. *How a sceptic attacks it:* familiar-farm validation, selection on the same folds, single seeds, empty-dominated MAE, a 37-image dense band, a custom metric, owner-drawn rotated boxes, a small test farm. Each is now addressed or stated: pre-registration, paired tests, seeds, non-empty MAE, band intervals with widths, blinded rotated boxes with a repeat, and an honest limitations list.

## Changes after review
- Added the CPU stage: root cause first (C1), hypotheses H7 onwards (C2), pre-registration of candidates, selection rule, bands and width cap (C3), training-data hygiene (C4), raw-image exclusion rules (C5), SAM centre check and flag handling (C6), blinded rotated subsets including the test farm (C7), scoring on Kaggle (C8), new tests (C9).
- Training schedule: convergence check (G0); finalists at the full schedule unless 72 epochs is shown to be enough. Model size M (and L if justified) added (G4).
- New-farm work led by the root cause, with GSD normalisation (G2) and leave-one-farm-out as a selection criterion, not only a report (G6, C3).
- Hard negatives mined by nested training (G3); known unlabelled-cattle images excluded from training (C4).
- Seeds for the reference and final model; paired bootstrap comparisons (G1, C3).
- Oriented boxes reframed as an output requirement with a written decision rule; fair budgets (authors' schedule per model); named fallback families (G7).
- Density head objective and ablation defined (G8); cross-conformal fitting, band-wise intervals with widths (G9); resolution study redesigned (G11).
- Compute ledger with priorities and a cut rule; timeline corrected to 7–11 weeks.
- Out-of-scope list added; the centre-placement claim softened (κ includes two people's noise, so some headroom may exist; G0 tests it indirectly).
- Pretraining licence policy raised as an owner decision (below).

## Owner decisions (2026-11-21)
1. **Compute:** rent an RTX 4090 on RunPod Community Cloud, prepaid $30 cap, automatic top-up off (D-35). Card order: Community RTX 4090 or 3090 first; if none is free, wait rather than pay more; Community RTX 5090 only for short jobs; Secure Cloud last. Small jobs (onboard model, density head, mining runs) go to free Kaggle. Cost is reported to the owner at $15 spent, and work stops for the owner's decision at $25.
2. **Starting weights:** option (a), allowed in Stage A and recorded in `docs/licences.md`, legal check before any product (D-36).
3. **Owner time:** about 7–8 hours of checking and drawing, agreed.
4. **Lean plan:** one combined recipe first; repeat runs only for close decisions; leave-one-farm-out for the reference and final model only; one oriented family against upright D-FINE; the resolution study reuses the combined-recipe model. The three folds stay for every candidate. Each shortcut and what it gives up is listed in the run ledger above.

## Verification
*(Filled in as each step completes.)*

- **C1, why Derval fails** (`scripts/derval_rootcause.py`, `results/baselines/derval_rootcause.json`):
  - **Not image scale, camera or height.** Matched animal for animal on size in pixels and on the brightness of the surroundings, Derval still misses 64% of its cattle where similar animals on other farms are missed 14% of the time. Derval misses at least 50% in every size band with 20 or more animals, against at most 14% elsewhere above 40 px. Mauron's flights with the same M3T camera at 100 m miss 17%, and Derval's 60 m images have a finer ground sample distance (2.2 cm) than those Mauron flights (3.6 cm).
  - **Contributing:** Derval's cattle are packed together (64% touch another animal, against 19% elsewhere), mostly dark (median share of very dark pixels 0.51 against 0.11), and a quarter are labelled although only partly visible (under 1.2 m long; 96% of those are missed).
  - **A large farm effect remains:** even Derval animals that are apart and not dark are missed 42% of the time (55 animals). With three training farms, a new farm's look (cattle, pasture, light) is the main risk; augmentation can reduce it, and only more varied training data can remove it.
  - **Effect on the plan:** G2's combined recipe leads with appearance changes (brightness and contrast towards dark animals and shade, colour jitter, greyscale, shadow), crowding (oversampling tiles with touching animals, and pasting animals into groups), and keeps only mild scale jitter; GSD normalisation moves to the reserve list. H7 is restated accordingly before any run.
- **C2, C3 (2026-11-21):** hypotheses H7–H12 written in `docs/hypotheses.md`; candidates, selection rule, interval settings, hypothesis thresholds and the G0 hold-out group recorded in `configs/phase2.yaml` and frozen (`results/frozen/phase2_prereg.json`) before any Phase 2 model was trained.
- **C4:** `scripts/make_empty_review.py` lists the 71 empty-labelled images where a Phase 1 model is confident it sees cattle, for the owner's review. Until the review is done, training leaves out the two images known to hold unlabelled herds (`results/data/training_exclusions.json`).
- **RunPod test sessions** (`results/phase2/run_ledger.csv`): the session chain works end to end (code from the private repository, tiles from Kaggle in under a minute, training, export, upload of outputs, self-deletion). An RTX 5080 trained 2.5 times as fast as Kaggle's T4 with 4 data-loading processes, with the GPU often waiting for data; Phase 2 runs use 12. Downloading the full ICAERUS archive from Zenodo ran at 2.8 MB/s.
- **Training code:** `gpu/train_dfine.py` (size, epochs, recipe, exclusions, touching-tile repeats, training-side hold-out) and the augmentations in `gpu/dfine_ops.py`, checked by a geometry test (rotations and flips keep every box exactly on its pixels) and a CPU run through training and export; `gpu/eval_tiles.py` scores a model on tiles with the project's scoring code.
- **G0, does 72 epochs converge?** (`gpu/runpod/job_g0.sh`, session g0-7, `results/phase2/g0_summary.json`, `g0_e72_eval.json`, `g0_e220_eval.json`): fold 0 with Jalogny/20230915 held out (548 hold-out tiles). 220 epochs gave no gain: centre-matched recall 0.878 → 0.866, precision 0.898 → 0.906, centre AP 0.721 → 0.711, upright AP50 0.980 → 0.971. H9 not held; every later candidate trains at 72 epochs (D-22), and the final model's full-schedule run (+35 to +55 T4 hours in the run ledger) is dropped. Cost $1.30 on a rented RTX 4080 SUPER.
- **G3, mining models and mining** (`gpu/runpod/job_g3.sh`, session g3-1; `scripts/mine_hard_negatives.py`): three mining models, each trained on one fold's tiles for 36 epochs (D-34), $0.50 on a rented RTX 5080. Mining ran on the laptop's processor. A first pass over a fixed half of the cross-validation photos gave too few tiles (185–288 per mining source against the 349–428 each fold needs from each source), so the remaining photos were mined too; the ratio was not lowered, because it was fixed before the run (D-34). With all 1321 photos, every source reached the number needed (374–492 tiles selected per source, 2179 distinct tiles to review); two sources have less than the planned 15% spare for tiles the reviewer rejects, and any shortfall after review is reported, not filled with unreviewed tiles. Next: the owner's review, then `apply`.
- **C4 review done (2026-11-22):** the owner answered all 71 photos: 3 cattle (the two known D-32 photos and one more), 3 not sure, 65 no cattle. The 3 "cattle" and the 3 "not sure" photos are left out of training (`results/data/training_exclusions.json`, now 6 photos); leaving a photo out removes a little training data and never changes scoring, so doubtful photos are left out. Hard-negative tiles cut from these photos are dropped too and replaced from the reviewed spares.
- **G3 review done (2026-11-22):** the owner marked 4 of the 2179 tiles as showing an animal (cattle in tree shade); a check of the highest-scoring unmarked tiles found silage bales and white sacks, as intended.
- **G2 prepared (2026-11-22), before any G2 model was trained:** the recipe is written down in `configs/g2_recipe.yaml` (touching tiles twice, copy-paste groups, mild scale jitter, the appearance ops, the reviewed hard negatives, the 6 excluded photos). New code: `HsCopyPaste` and `HsScaleJitter` in `gpu/dfine_ops.py`, checked on real tiles (every pasted animal has a box, no labelled animal is more than 30% covered, boxes stay inside the tile); `--hardneg` in `gpu/train_dfine.py`, which refuses any hard negative from the fold's own images or from an excluded photo (`tests/test_train_dfine.py`); a CPU run of the full G2 recipe through training and export. The reference recipe is retrained in the same session on the same GPU type (D-35). The hard-negative set is packaged as `herdsight_hardneg_v1` (1958 tiles; manifest `results/phase2/herdsight_hardneg_v1_manifest.json`), checked against the repository manifest at the start of the session.

- **G2 result** (`scripts/phase2_compare.py`, `results/phase2/g2_comparison.json`; six 72-epoch models, full-image out-of-fold predictions of 1,323 images per recipe made on one GPU). The combined recipe does **not** replace the reference. Centre AP 0.5453 against 0.5446 (difference +0.0008, -0.0094 to +0.0088); count MAE 0.533 against 0.571 (-0.196 to +0.063). Both intervals cross zero. H7 and H8 are not held; Derval recall stays at 0.17 against 0.15. Derval is one farm-date group, so its interval from group resampling carries no information and is not quoted. Under the frozen rule this is a close call, for which repeat runs (seeds 1 and 2) would come before a final decision. The owner decided on 2026-11-23 not to run them (about $6), keeping that budget for the new-farm problem. This is a departure from G1: the G2 comparison has no noise floor from repeat seeds, so it does not show that the two recipes perform the same. The reference stays the best recipe, and G4 (D-FINE M) runs on it (session g4-1, RTX 3090). G4 and G6 sessions use the GPU types that had free machines (RTX 3090, RTX 5080), a departure from D-35's single GPU type, to be stated beside their comparisons.
- **G2 flight-count check** (`scripts/score_s3.py`; `results/phase3/dfine_g2_s3_v20.json`, `dfine_ref_s3_v20.json`; Phase 1 linking on the 20 verified flights). Mean flight count error 23.1% for the G2 recipe against 26.7% for the same-session reference; difference -3.6 points (95% interval -13.1 to +3.7). The interval crosses zero and individual Mauron flights swing widely between the two models, so this check is reported only and is not used to select a model. These are public-data results, not field results.
- **G4 and G6 training** (sessions g4-1, g6ref-3). D-FINE M on the reference recipe (three folds, 72 epochs, RTX 3090) and the three leave-one-farm-out models of the reference recipe (72 epochs, RTX 5080) are trained and their ONNX hashes verified. Full-image predictions were made on one RTX 3070 (session g4g6-score-4). Tile-level monitor figures are progress checks only.
- **G4 result** (`results/phase2/g4_comparison.json`; D-FINE M, reference recipe, 1,323 images). D-FINE M does **not** replace D-FINE S. Centre AP 0.5543 against 0.5446 (+0.0097, -0.0074 to +0.0292); count MAE 0.579 against 0.571 (+0.008, -0.113 to +0.137). Both intervals cross zero, and the centre AP gain is below the 0.03 that H12 requires without repeat runs, so H12 is not held. Derval recall is 0.23 against 0.15; Derval is one farm-date group. Repeat runs have not been made. The two models were trained on different GPU types, a departure from D-35.
- **G6 result, reference recipe** (`scripts/phase2_lofo.py`, `results/phase2/g6_lofo.json`). Each farm is predicted by a model trained only on the other two, with its operating threshold set on those other farms. Pooled centre AP 0.397 against 0.545 under the official cross-validation (-0.147, -0.204 to -0.089); recall 0.553 against 0.728 (-0.175, -0.250 to -0.098); count MAE 1.098 against 0.571 (+0.526, +0.278 to +0.783). Recall by farm, unseen against official: Derval 0.21 against 0.15; Jalogny 0.72 against 0.87; Mauron 0.46 against 0.68. The official figures therefore depend substantially on having seen other days of the same farm. The leave-one-farm-out models also trained on less data, so the drop is not due to the new farm alone. Three farms from one country are a small sample. Public-data results, not field results.
- **C5 result** (`scripts/c5_raw_images.py`, `results/phase2/c5_raw_images.json`). All 884 ICAERUS raw images were sorted by file name, date and GPS, without decoding any pixels. 815 are already in the labelled dataset and the other 69 come from a farm and day already in a cross-validation fold. None is linked to the test farm and none comes from a new place or day, so no image qualifies and this step is dropped, as the plan allows.
- **G11 result** (`scripts/resolution_score.py`, `results/phase2/g11_resolution.json`; G2 models, 1,323 images, public data with simulated degradation). Recall was 0.715 at native resolution. With blur, noise and JPEG compression at full size it was 0.708 (-0.006, -0.014 to +0.003). At 75% scale it fell to 0.647 (-0.068, -0.116 to -0.014), past the 0.05 allowance, and at 50% to 0.481. The smallest passing scale is therefore full size, so the provisional budget of 35.4 pixels for the smallest animal stands and the survey cannot fly higher on this evidence. Scales between 75% and 100% were not tested. Measured on adult cattle: this is a lower bound for calves, not a calf result and not a field result.
- **G7 result** (`scripts/phase2_compare.py`, `results/phase2/g7_comparison.json`; three RTMDet-R models on the authors' 36-epoch schedule, full-image out-of-fold predictions of 1,323 images). RTMDet-R is clearly worse than the reference D-FINE: centre AP 0.218 against 0.545 (-0.326, -0.383 to -0.279), recall 0.519 against 0.728, count MAE 0.745 against 0.571, Derval recall 0.090 against 0.146. By the rule written before the run, the upright detector keeps the counting and an angle is to be predicted for each counted animal (D-48, proposed). RTMDet-R started from an ImageNet backbone while D-FINE started from a COCO detector, which may explain part of the gap; no extra tuning was done.
