# Decision log

Every design decision, the options considered, and why the choice was made. D1–D17 are the starting decisions; each may be replaced by a better option, which is then logged with its evidence.

Format:

```
## D-<number>: <title>  (<date>)
Options considered: ...
Choice: ...
Evidence / reasoning: ...
Easy option rejected (if any) and why: ...
Status: proposed | adopted | superseded by D-<n>
```

---

## D-1: New repository on a shared core library  (2026-11-18)
Options considered: fork drone-ranger; retrain drone-ranger's detector; new repository depending on a core library extracted from drone-ranger; vendor drone-ranger modules.
Choice: New repository that depends on a pinned version of a core library (ground projection and geometry, navigation filter, tracker, conformal calibration, multi-drone communication, ONNX runtime wrappers, ROS frame conversions) extracted from drone-ranger. If extraction is impractical, vendor the needed modules with attribution and log why.
Evidence / reasoning: Counting needs duplicate removal, motion handling, a herd register and alerts that drone-ranger lacks; a shared library keeps one tested copy of the geometry.
Easy option rejected (if any) and why: Forking or retraining drone-ranger would inherit its pedestrian-trained detector and down-scaled processing, the main causes of its poor recall at height.
Status: adopted; reuse method refined by D-20

## D-2: Specifications before code, baseline before targets  (2026-11-18)
Options considered: set targets up front; set targets after an honest baseline.
Choice: Run a quick honest baseline for each specification, then set an ambitious but achievable target, then freeze the specification file (record its hash) before the main work.
Evidence / reasoning: Targets set without a baseline are guesses; freezing prevents moving the goalposts.
Easy option rejected (if any) and why: Inventing numbers before measuring.
Status: adopted

## D-3: Ground truth and evaluation protocol  (2026-11-18)
Options considered: upright-box Intersection over Union (IoU) only; centre matching with a similarity score; rotated IoU against generated labels.
Choice: Three complementary scores: centre matching with an Object Keypoint Similarity (OKS)-style centre score as the main counting score; upright-box AP50 and AP50:95 for comparison with published work; rotated IoU on a small hand-verified subset of touching and overlapping animals. Splits by dataset, flight, farm or date, never by image.
Evidence / reasoning: Rotated predictions scored against upright labels look wrongly poor under IoU; centre matching is orientation-independent. Images from one flight are near-duplicates.
Easy option rejected (if any) and why: Scoring against labels we generated ourselves.
Status: adopted

## D-4: Public data first, loaders ready for real flights  (2026-11-18)
Options considered: wait for farm data; synthetic data only; public data with loaders that also accept real flights.
Choice: Public datasets first. Every loader accepts real farm flights (geotagged stills with EXIF, flight logs) with no code changes.
Evidence / reasoning: No farm access in Stage A; work must carry into Stage B without being thrown away.
Easy option rejected (if any) and why: Loaders tied to one dataset's quirks.
Status: adopted

## D-5: Sensors  (2026-11-18)
Options considered: RGB only; thermal only; aligned RGB plus thermal.
Choice: Design for a high-resolution RGB stills camera (at least 20 MP, mechanical or global shutter) plus an aligned radiometric thermal camera (640×512). Stage A trains on RGB (most data available) and adds thermal where labelled data allows (BIRDSAI, ICAERUS thermal). The pipeline stays multi-sensor ready.
Evidence / reasoning: Theft often happens at night, which needs thermal.
Easy option rejected (if any) and why: RGB-only design, which would need rework for night surveys.
Status: adopted

## D-6: Flying height from a pixel budget  (2026-11-18)
Options considered: fixed height; height adjusted continuously; height computed once per mission from the pixels needed for the smallest animal.
Choice: Height above ground level (AGL) = focal length in pixels × required Ground Sample Distance (GSD). Required GSD is set by the smallest animal that must be counted (calves; later lambs), measured from data with a resolution study. Height is computed once at planning time; the drone flies level at constant AGL over a smoothed terrain model. Welfare limits encoded in the planner: survey at 50 m or higher; herd mode at 30–40 m or higher; approach at height, never descend onto animals; prefer zoom to descending.
Evidence / reasoning: drone-ranger's recall fell from 47% at 20 m to 6% at 50 m because targets were too few pixels. Evidence suggests animals react below roughly 20–40 m and habituate over repeated flights (sources to be cited in the write-up).
Easy option rejected (if any) and why: Assuming a pixel budget instead of measuring it.
Status: adopted

## D-7: Survey modes  (2026-11-18)
Options considered: video lawnmower sweep; planned geotagged stills; stills plus hovering over herds.
Choice: (1) Camp survey: geotagged stills along a planned route with enough overlap for duplicate removal. (2) Herd mode: when onboard herd detection finds a group, hover above it (within D-6 limits) to capture it in one or a few frames. (3) Identification pass: later research.
Evidence / reasoning: Stills give full resolution and geotags; herd mode removes most double counting caused by movement.
Easy option rejected (if any) and why: Video sweep, which gives lower resolution and many near-duplicate frames.
Status: adopted

## D-8: Detection model  (2026-11-18)
Options considered: upright boxes; Oriented Bounding Boxes (OBB); segmentation; density maps.
Choice: OBB as the primary output, run at full resolution with overlapping tiles at home base. Architecture and size chosen by held-out accuracy first, then speed, comparing at least two strong model families. A density-map counting branch as a cross-check: where it disagrees with the detection count beyond a threshold set on validation, mark the cluster as dense and widen its interval (and trigger herd mode in simulation and Stage B). Segmentation later, for herd mode.
Evidence / reasoning: Cattle are elongated and lie at angles in packed groups; upright boxes overlap heavily and non-maximum suppression deletes real animals.
Easy option rejected (if any) and why: Down-scaling images to speed up inference.
Status: adopted

## D-9: Counting across images by global matching on the map  (2026-11-18)
Options considered: greedy pairwise matching; global assignment across all images.
Choice: Project each detection to the ground with a position covariance; build candidate matches between overlapping images using covariance-weighted distance, an appearance embedding and an animal-motion model; solve one global assignment (graph optimisation / min-cost flow / joint Hungarian); report the unique count with a calibrated conformal interval, e.g. "143 (141–145 at 90%)". Compare with the published duplicate-removal method on the Brazilian geolocated benchmark.
Evidence / reasoning: Greedy pairwise matching causes chain errors.
Easy option rejected (if any) and why: Greedy pairwise matching.
Status: adopted

## D-10: Where processing happens  (2026-11-18)
Options considered: all onboard; all at home base; light onboard detection plus full processing at home base.
Choice: Stage A: the full pipeline runs at home base (a ground station with a GPU) on full-resolution stills after landing. Onboard (simulated in Stage A): only light herd detection to trigger herd mode, sized for Raspberry Pi 5 class compute and timed on real hardware if available. Future plan: an onboard GPU (Jetson Orin Nano Super class) for counting in flight.
Evidence / reasoning: Streaming full-resolution stills over a drone radio is impractical. ARM emulation is not a valid speed measurement.
Easy option rejected (if any) and why: Shrinking the model to fit onboard compute.
Status: adopted

## D-11: Simulation instead of a drone in Stage A  (2026-11-18)
Options considered: no flight testing; PX4 Software-In-The-Loop (SITL) with Gazebo Harmonic and ROS 2 Jazzy.
Choice: A simulated drone flies the real mission code over simulated camps with cattle models placed in realistic groups (group sizes and spacing estimated from public data) that move during surveys (grazing / collective-motion model). Simulation is used for geometry, duplicate removal, mission logic and alerts, never for detector-accuracy claims.
Evidence / reasoning: No drone in Stage A; simulation gives exact ground truth by construction.
Easy option rejected (if any) and why: Static, evenly spaced cattle that never move.
Status: adopted

## D-12: Identity  (2026-11-18)
Options considered: tags; coat-pattern identification; vision-only counting.
Choice: Vision-only counting. An empty `IdentitySource` interface is left so tags can be added later as a separate product. Coat-pattern identification is a later research track.
Evidence / reasoning: No tags in Stage A; counting does not require individual identity.
Status: adopted

## D-13: Alerts  (2026-11-18)
Options considered: alert on point count below expected; alert only when the whole interval is below expected, with recounts.
Choice: Keep an expected count per camp (the herd register), editable for births, sales and moves. Alert only when the whole calibrated interval is below the expected count. Recount before alerting (second survey or herd mode); require two independent surveys to agree unless the shortfall is large (threshold to be set). The alert threshold is a dial, with a trade-off curve (false alarms per 100 surveys against share of real losses caught) produced in simulation.
Evidence / reasoning: False alarms erode trust and must be kept to a minimum.
Easy option rejected (if any) and why: Alerting on the point count alone.
Status: adopted

## D-14: Python and C++  (2026-11-18)
Options considered: Python only; C++ only; Python for development and ML, C++ for onboard and shared core.
Choice: Python for training, evaluation, experiments, data tools and the dashboard backend. C++20 (CMake, Eigen, OpenCV, GoogleTest, address and undefined-behaviour sanitizers in CI) for onboard ROS 2 nodes and the shared geometry and matching core, with pybind11 bindings so evaluation calls the same code that flies. Every algorithm has a Python reference and a parity test against the C++; algorithms move to C++ once they stop changing.
Evidence / reasoning: Each language used where it is strongest; parity tests prevent the two copies drifting apart.
Status: adopted

## D-15: Embedded C  (2026-11-18)
Choice: Postponed together with tags (D-12).
Status: adopted

## D-16: ROS 2 runs the mission  (2026-11-18)
Options considered: ROS 2 as a thin wrapper; ROS 2 controlling the drone.
Choice: C++ composable lifecycle nodes with intra-process zero-copy (camera → herd detector → geolocator → mission controller). PX4 through uXRCE-DDS (`px4_msgs`), with offboard setpoints / mission upload. Actions `SurveyCamp(camp_id)` and `InspectHerd(position)`. BehaviorTree.CPP for mission logic (survey → herd found → herd mode → resume → battery low → return to land). TF2 for all frames including the gimbal. `launch_testing` integration tests and rosbag regression tests in CI. Counting itself runs at home base.
Evidence / reasoning: In drone-ranger, ROS 2 was a thin wrapper and nothing controlled the drone.
Easy option rejected (if any) and why: A wrapper that only republishes detections.
Status: adopted

## D-17: Dashboard for non-technical farmers  (2026-11-18)
Options considered: technical dashboard; farmer-first dashboard built on a versioned results schema.
Choice: Results schema first (versioned JSON per survey: farm, camp, time, count, interval, expected count, status, alerts, evidence images); the dashboard reads only this. A glossary file gives every technical term one plain-language version, ready for translation. Screens: farm map with camps coloured green / amber / red; camp page; alerts with plain sentences, photos and actions; herd records. Phone-first, works on weak signal, shows last results offline. Suggested stack: FastAPI + PostgreSQL/PostGIS backend and a Progressive Web App (PWA) frontend with MapLibre (may be revised, with reasons logged). A mock-data generator lets the dashboard be built before the pipeline is finished. Usability tested against S6.
Evidence / reasoning: The user is a farmer; technical wording must never reach them.
Status: adopted

## D-18: Training compute  (2026-11-18)
Options considered: local GPU (none available); paid cloud GPUs; free cloud notebooks (Kaggle, Google Colab).
Choice: Kaggle notebooks first, Google Colab as fallback. Training code runs as a script driven by a config file, so the same command runs on Kaggle, Colab or a local machine; notebooks only call the script. Weights and run outputs are downloaded back to external storage after each run.
Evidence / reasoning: No local NVIDIA GPU and no budget for paid compute (§6). Free tiers have session and weekly time limits, so training must checkpoint and resume.
Easy option rejected (if any) and why: Shrinking images or models to fit the free time budget. Full-resolution tiles stay; if the free budget is not enough, this is raised as a question rather than cutting accuracy.
Status: adopted

## D-19: Operating countries and flight rules  (2026-11-18)
Options considered: design for one country; per-country rule profiles.
Choice: Mission planning reads a per-country rules profile (maximum height, line-of-sight requirements, distance limits) for the United Kingdom and South Africa. The planner refuses a mission that breaks the active profile. Exact limits are taken from the current UK Civil Aviation Authority (CAA) and South African Civil Aviation Authority (SACAA) rules, with citations, in `configs/`.
Evidence / reasoning: The owner is in the UK; South African farms face high stock theft and have much larger camps, which affects survey height, route length and battery planning.
Easy option rejected (if any) and why: Hard-coding one height limit.
Status: adopted

## D-20: Reuse drone-ranger by copying selected modules, not a shared library (refines D-1)  (2026-11-18)
Options considered: separate shared library repository pinned by both projects; depending on drone-ranger directly at a pinned commit; copying (vendoring) selected modules with provenance.
Choice: Copy the geometry (frames, camera, ranging, ground covariance, terrain, uncertainty, Kalman filter), evaluation (conformal intervals, statistics) and utility (config, seeds, environment report) modules, and the document-filling script, from drone-ranger commit b4f4e4f, with their tests. Each file names its source file and commit; changes beyond import paths are listed in `docs/plans/phase0-setup.md`.
Evidence / reasoning: drone-ranger is finished and its published results come from frozen code, so refactoring it around a new library would change the code behind those results. A library with one active user adds release work for no gain, and under D-14 the geometry used in production moves to C++, leaving the Python copy as the reference implementation. Depending on drone-ranger directly would pull in its unrelated dependencies (tracking metrics, HDF5, ONNX Runtime). Not taken: navigation filter, frame-to-frame tracker, search and multi-drone code, ONNX detector wrappers (reasons in the Phase 0 plan).
Easy option rejected (if any) and why: Copying the whole package and deleting what fails, which would bring untested, irrelevant code along.
Status: adopted

## D-21: Dataset licence policy  (2026-11-18)
Options considered: use every public dataset for training; use non-commercial data for training while the project is non-commercial; train only on data whose licence allows commercial use.
Choice: Training uses only data whose licence allows commercial use (for example CC BY 4.0). Non-commercial data (for example CC BY-NC-ND) is used for evaluation only, and marked as such in every result. Data with no stated licence is not used until its authors give written permission.
Evidence / reasoning: The repository is all rights reserved and the project may become a product (brief §1, §6). A model trained on non-commercial data could not be sold without retraining, which would invalidate every result measured on it. Keeping training data commercially clean from the start means Stage A results carry into Stage B (brief §6).
Easy option rejected (if any) and why: Training on everything available now and sorting out licences later. That would give more training data today, but results that cannot carry over to a product.
Status: adopted 2026-11-19 (docs/questions.md Q9)

## D-22: Phase 1 fine-tuned baseline recipe  (2026-11-18)
Options considered: the authors' full D-FINE S custom-dataset recipe (220 epochs, batch 64); a shortened recipe that fits one free Kaggle session; a smaller model.
Choice: D-FINE S, started from the authors' COCO weights, on 640 px tiles cut at native resolution from the training flights only; total batch 16 with learning rates scaled linearly from the authors' 64; 72 epochs, with augmentation off for the last 6; seed 0; the last checkpoint is kept, with no checkpoint selection on any split. Every tile containing animals is used, plus one randomly drawn empty tile for each (seed 0). Script: `gpu/train_dfine_baselines_kaggle.py`.
Evidence / reasoning: A baseline has to be honest and repeatable, not optimal. It sets the floor that Phase 2's architecture comparison must beat. 220 epochs at batch 16 does not fit a 12-hour free session. Keeping the last checkpoint avoids choosing a model on data that is later used for calibration or reporting.
Easy option rejected (if any) and why: Down-scaling whole images to 640 px to train faster. That repeats drone-ranger's main mistake (brief §4), so tiles keep native resolution. The shorter schedule is recorded as a limitation of the baseline only; Phase 2 trains properly.
Status: adopted

## D-23: Merge tile detections by centre ownership, not non-maximum suppression  (2026-11-18)
Options considered: non-maximum suppression (NMS) across overlapping tiles; weighted box fusion; centre ownership (each tile keeps only detections whose centre lies in its own share of the overlap).
Choice: Centre ownership. Owned regions split each overlap at its midpoint and together cover the image exactly once (tested). The overlap is set from the measured 99.9th-percentile animal length, so every animal lies wholly inside at least one tile.
Evidence / reasoning: NMS removes overlapping boxes, which deletes real animals standing close together. That is the failure oriented boxes are meant to fix (D-8), and it must not be reintroduced at the tiling step. Centre ownership has no free threshold to tune.
Easy option rejected (if any) and why: NMS with a default IoU threshold, which is the common practice for tiled inference.
Status: adopted

## D-24: Zero-shot baseline on a fixed image sample  (2026-11-18)
Options considered: zero-shot on every calibration and validation image on the laptop CPU (more than 30 hours); on a Kaggle GPU (needs the images uploaded); on a fixed random sample on the CPU.
Choice: OWLv2 (no cattle training) on a seeded random 25% of calibration and of validation images, on 960 px native-resolution tiles. The fine-tuned baseline is scored on all images, and also on exactly this sample, for the comparison in H1.
Evidence / reasoning: Measured 7 seconds per 960 px tile on the laptop CPU (about 4 minutes per image). The sample is drawn by a fixed rule before any result exists, so it cannot favour either method.
Easy option rejected (if any) and why: Shrinking images so the zero-shot model runs faster, which would understate what it can do.
Status: adopted

## D-25: Split by farm and survey date, not by single flight  (2026-11-18)
Options considered: one flight as the unit; farm and survey date as the unit; whole farms only.
Choice: Flights over the same farm on the same date stay in the same split. The test farm is still a whole farm.
Evidence / reasoning: The first splits file (hash `5916c3750d6049ae…`), made by flight, put flights flown minutes apart over the same herd into different splits (for example Jalogny `DJI_202309151045_055` in validation, `…_056` in training). Those images are near-duplicates, the leak the brief warns against (§3.2, §8.2). It was replaced before any model was run.
Easy option rejected (if any) and why: Keeping flight-level splits, which would make validation look better than it is.
Status: adopted

## D-26: Grouped three-fold cross-validation for Phase 1 baselines  (2026-11-18)
Options considered: one train / calibration / validation split; grouped cross-validation over the farm-date groups.
Choice: The 10 farm-date groups outside the test farm are dealt into 3 folds, balanced by box count (`results/splits.json`). Each fold is predicted by a model trained only on the other two. For each fold, the operating threshold and the conformal count intervals are fitted on the other folds' out-of-fold predictions, then applied to that fold. Intervals are bootstrapped over the 10 groups. The test farm (Other_farms) is untouched until Phase 7.
Evidence / reasoning: With date grouping, a single 60/20/20 split (hash `b9d6097fc59c513e…`, replaced before any model was run) left validation as one farm on one day: 2 flights and 422 boxes. That gives no meaningful uncertainty. Cross-validation scores every non-test image once with a model that has not seen its farm and day.
Easy option rejected (if any) and why: Keeping the single split because it needs one training run instead of three.
Status: adopted

## D-27: Register overlapping images from their content before matching animals  (2026-11-18)
Options considered: match animals across images using geotag projection only (D-9 step 1 as written); register each overlapping pair of images from image features, using the geotag as the starting point and as a gate.
Choice: Register. Pasture is close to flat at survey scale, so a homography between overlapping nadir stills, fitted by RANSAC to feature matches, maps one image's pixels into the other's. Animals are then matched in registered pixel space, with the geotag projection as a sanity gate. The global assignment (D-9 step 3) is unchanged.
Evidence / reasoning: Measured on ICAERUS, geotag projection puts the same animal a median 1.5 m apart, and 6.4 m at the 90th percentile, in consecutive images (`results/data/geolocation_check.json`). Cattle in packed herds stand 2 to 3 m apart, so geotags alone would merge or split neighbours. Moving animals and hills limit registration too; where registration fails, the pair falls back to the geotag with its larger covariance, and this is reported.
Easy option rejected (if any) and why: Geotag-only matching, which is simpler but fails in exactly the packed herds that matter most for counting.
Status: adopted

## D-28: What counts as a herd for onboard herd detection (S7)  (2026-11-18)
Options considered: any detected animal; a fixed count within a fixed area; single-linkage groups at a measured spacing.
Choice: A herd is a group of at least 3 animals in which each animal is within 10 m of another (single linkage on ground positions). S7's herd recall is the share of such herds in which the onboard detector finds at least one animal.
Evidence / reasoning: The 95th percentile of the nearest-neighbour distance between cattle is 11.5 m (`results/data/herd_stats.json`). At 10 m, 87% of animals fall in groups of 3 or more. Herd mode's purpose is to stop over a group and capture it in a few frames (D-7), so a single straggler need not trigger it. A lone animal is still counted by the camp survey itself.
Easy option rejected (if any) and why: Counting any single detection as a herd, which would make herd recall look higher than it is.
Status: adopted 2026-11-19

## D-29: Onboard herd detector input for S7  (2026-11-18)
Options considered: the home-base detector on full-resolution tiles onboard; a small detector on reduced images.
Choice: D-FINE N trained on 640 px tiles of images reduced to a 1,280 px long side (6 tiles per Mavic 3E still), by the same three-fold cross-validation. Latency is reported as not measured until it is run on a Raspberry Pi 5; the input size is an estimate to be checked then.
Evidence / reasoning: A full-resolution still is 108 tiles. At drone-ranger's estimated 285 ms per D-FINE N tile on a Pi 5 (an estimate, not a measurement), that is about 30 s per still, far slower than the few seconds between survey stills. Reduced to a 1,280 px long side, a still is about 2 s. Animals then shrink to roughly 20 px long, which is the cost S7 measures.
Easy option rejected (if any) and why: Reporting S7 latency from a laptop or under ARM emulation, which D-10 rules out as a speed measurement.
Status: adopted 2026-11-19 with the input size as an estimate: no Raspberry Pi is available (docs/questions.md Q8), so S7 latency is reported as an estimate from published or drone-ranger figures, labelled as such, and never as a measurement

## D-30: Group animal links under a one-per-image constraint  (2026-11-18)
Options considered: connected components of pairwise links; greedy shortest-first grouping that never puts two detections of one image in one identity; global assignment (D-9 step 3).
Choice: Greedy constrained grouping for the Phase 1 candidate identities and as the duplicate-removal baseline. Global assignment comes in Phase 3, and must beat it.
Evidence / reasoning: Plain connected components chained neighbours together. In flight Mauron/DJI_202311071416_061 they gave 2 animals for 142 labels, although one image alone shows 12 different cattle. With the constraint, every flight's count is at least its largest single-image count (14 for that flight; `results/data/s3_candidate_links.json` lists both).
Easy option rejected (if any) and why: Connected components, the usual simple choice, which fails exactly in packed herds.
Status: adopted

## D-31: How the Phase 1 targets are set from the baselines  (2026-11-20)
Options considered: a fixed rule applied by a script to the best baseline (close half the gap to perfect for scores, halve errors); targets set by judgement after reviewing the baseline results and where the baselines fail.
Choice: Targets are set by the owner after reviewing the scored baselines and their failure cases, each with a stated reason, in `configs/specs.yaml`. `scripts/baseline_summary.py` first gathers, for each measure, the best Phase 1 baseline, its 95% interval and its source file (`results/baselines/spec_baselines.json`), with the half-the-gap value shown as a reference only. The targets file is frozen with its hash (`scripts/freeze.py phase1_specs`) before any Phase 2 model is trained or scored. The 90% count-interval coverage target stays 0.85–0.95, as the brief sets, and S7 latency is the D-29 estimate, marked as not measured.
Evidence / reasoning: Before seeing the baselines it is not known which errors dominate or how much room there is to improve, so a single rule could set targets that are meaningless for some measures. What keeps the targets honest is that they are fixed, with reasons, before the system they judge exists.
Easy option rejected (if any) and why: Leaving the targets open until Phase 2 results are in, which would invite targets the new system just meets.
Status: adopted 2026-11-20 (owner); replaces the fixed-rule proposal of 2026-11-19

## D-32: Keep the published ICAERUS labels for scoring  (2026-11-21)
Options considered: score against the published labels unchanged; correct the images found with unlabelled cattle and rescore every baseline; re-check every image labelled empty by hand, without looking at any model's output.
Choice: Score against the published labels unchanged. Images found to contain unlabelled cattle are listed in `docs/data.md` as a known data issue.
Evidence / reasoning: The failure review (`results/baselines/dfine_failures.json`) found at least two images in Jalogny flight DJI_202309261339_058 whose label files are empty although they show a herd. Both were found because the fine-tuned detector fired on them. Correcting only the images a model flags would remove that model's false detections and not the errors other models make, so it would favour the model that found them. Leaving them in can only make the reported precision and false-detection rates slightly worse than the truth.
Easy option rejected (if any) and why: Fixing the two obvious images, which is quick but biased for the reason above.
Status: adopted 2026-11-21 (owner)

## D-33: Measure new-farm performance by leaving one farm out  (2026-11-21)
Options considered: rely on the frozen grouped cross-validation only; also score by leave-one-farm-out over the three cross-validation farms (Mauron, Derval, Jalogny).
Choice: Leave-one-farm-out is run for the reference recipe and the final Phase 2 model, and reported beside the official cross-validation with worst-group recall and count error. A candidate that loses on it is not chosen. The frozen cross-validation stays the measure for the targets.
Evidence / reasoning: In Phase 1 the only farm scored by a model that had never seen it (Derval) lost 67% of its cattle (`results/baselines/dfine_failures.json`). Most cross-validation groups share a farm with training data, so the official measure mostly tests familiar farms on a new day. UK and South African farms will be new farms.
Easy option rejected (if any) and why: Reporting only the pooled cross-validation, where Derval is 6% of the animals and its failure barely moves the totals.
Status: adopted 2026-11-21 (owner)

## D-34: Mine hard negatives by nested training  (2026-11-21)
Options considered: mine a fold model's false detections on its own training images; mine the Phase 1 out-of-fold false detections; nested mining (for fold k with training folds a and b, train on a and mine on b, then train on b and mine on a).
Choice: Nested mining. The mining models train at half the schedule, because they only find confusing objects and are never scored. Crops are confirmed by the owner as containing no animal, and added at one hard negative per two positive tiles, fixed before the run.
Evidence / reasoning: A model's false detections on its own training images are few and biased towards unlabelled cattle (D-32). Out-of-fold false detections come from models that trained on fold k, so fold k's labels would influence the negatives used when scoring fold k. Nested mining keeps every fold's labels out of its own negatives.
Easy option rejected (if any) and why: Reusing the Phase 1 out-of-fold false detections, which costs no GPU time but leaks fold labels into training.
Status: adopted 2026-11-21 (owner)

## D-35: Rent a GPU for Phase 2 training (refines D-18)  (2026-11-21)
Options considered: free Kaggle only (about 30 GPU hours a week); Colab Pro; renting an RTX 4090 by the hour (RunPod or Vast.ai); buying a GPU and a desktop.
Choice: RunPod Community Cloud, RTX 4090, on-demand, from prepaid credit of $30 with automatic top-up off. Each session is self-contained: it downloads the public data and the code, trains and scores, uploads the outputs to a private Kaggle dataset, and deletes itself only after the upload is confirmed. Kaggle stays available for small jobs. Every Phase 2 model is trained on the same GPU type, so comparisons are fair.
Evidence / reasoning: Phase 2 needs about 130–185 T4 GPU hours (plan run ledger), which is 5–7 weeks of free Kaggle quota. At $0.34 per hour (RunPod pricing page, 2026-11-12) and an estimated 3–5 times the T4's speed, a 4090 does this for about $9–20, with no session caps. The speed ratio is measured on the first session before further spending. Community Cloud pods cannot use RunPod's network volumes, so no data is kept between sessions; the data is public and re-downloads in minutes. A training desktop would cost about £1,000–1,500, and the brief defers hardware purchases to the go/no-go gates (§6).
Easy option rejected (if any) and why: Fitting Phase 2 into the free quota by shortening training or skipping comparisons, which costs accuracy (D-18 says to raise the question instead).
Status: adopted 2026-11-21 (owner)

## D-36: Pretrained starting weights (extends D-21)  (2026-11-21)
Options considered: allow starting weights pretrained on public research datasets (ImageNet, COCO, DOTA); allow only some of them; allow none.
Choice: Models may start from weights pretrained on public research datasets during Stage A. Every set of starting weights used is recorded with its source and licence in `docs/licences.md`. Before any product (Stage B) the starting weights get a legal check, and a model is retrained if needed. Training data itself stays under D-21.
Evidence / reasoning: Almost every detector, including the Phase 1 D-FINE models, starts from weights pretrained on research datasets with non-commercial terms; training from scratch on about 3,400 cattle would cost far more accuracy than any other choice in this project. Treating ImageNet, COCO and DOTA the same keeps the rule consistent.
Easy option rejected (if any) and why: Ignoring the question, which would leave an unrecorded licence risk in every model.
Status: adopted 2026-11-21 (owner)

## D-37: The S4 alert target, and more simulated farms to judge it  (2026-11-21)
Options considered: set the target from the 400-farm baseline; set it from a larger run of the same simulator; leave it open until Phase 3.
Choice: The validation and test sets of simulated farms are raised from 400 to 4000 farms each (`configs/sim_farms.yaml`), before any test-set run. The S4 target (`configs/specs.yaml`) is the recount rule at the 90% interval level with at most 1.5 false alarms per 100 surveys and at least 65% of losses of 5 or more animals caught within 3 surveys, measured with the count error model refitted to the latest measured flight errors. The Phase 4 hypotheses (H13-H15) keep the outcomes judged on the original 400 farms, as pre-registered; the 4000-farm run is reported beside them.
Evidence / reasoning: The 400 farms had only 22 losses of 5 or more animals, so the share caught had an interval of 23%-70% and could not separate a good rule from a poor one. 4000 farms give 212 such losses. On them the recount rule at 90% gives 1.45 false alarms per 100 surveys and catches 58% (51%-64%) of losses of 5 or more with the Phase 1 count error, and 1.17 and 67% (60%-73%) with the S3 target's count error (`results/phase4/s4_validation_4000farms.json`). A 65% target is met only if Phases 2 and 3 bring counting to about the S3 target, which ties S4 to the detector and linking work. The first 400 farms of the larger run are the baseline farms, and H13-H15 come out the same on 4000 farms. Filling the S4 entry changes `configs/specs.yaml`, which is part of the Phase 1 and Phase 2 freezes, so both are re-frozen; the earlier manifests are kept in `results/frozen/history/`, and no other value changed.
Easy option rejected (if any) and why: Setting 60% from the 400-farm run, which the larger run shows the current counting nearly meets already.
Status: adopted 2026-11-21 (owner delegated the choice)

## D-38: Keep the large-loss shortcut; S4 depends on counting accuracy  (2026-11-21)
Options considered: keep the recount rule's large-loss shortcut on the 90% interval; judge the shortcut on the 99% interval (H16); remove the shortcut.
Choice: Keep the shortcut as it is (recount rule, 90% interval, 10 animals). The way to reach the S4 target is better counting (Phases 2 and 3), not a different alert rule.
Evidence / reasoning: The S4 breakdowns (`results/phase4/s4_breakdown_validation.json`, 4000 simulated farms) show that 57% of the remaining false alarms come from count error in camps of 20 or more animals; with the shortcut removed, false alarms fall from 1.45 to 0.51 per 100 surveys, so most of them are raised by it. Judging the shortcut on the 99% interval cut false alarms from 1.45 to 0.62 per 100 surveys but the share of losses of 5 or more caught fell from 58% to 43% (H16 not held), and no dial setting then met the target (H17 not held): its curve lies on the current one. Removing the shortcut altogether also catches fewer losses of 5 or more. With the S3 target's count error, the current rule meets the S4 target (1.17 false alarms per 100 surveys, 67% caught). Simulation results, not field results.
Easy option rejected (if any) and why: Adopting the stricter shortcut for its lower false-alarm rate, which would hide a loss of a quarter of the large thefts caught.
Status: adopted 2026-11-21 (owner delegated the choice)

## D-39: The history rule as the alert rule for the cautious end of the dial  (2026-11-22)
Options considered: alert from the latest survey with a recount (the recount rule, the S4 reference); judge shortfalls from every survey since the camp's records were last confirmed (the history rule, a Kalman filter of the camp's shortfall; Phase 4 plan stretch candidate); offer both on the dial.
Choice: The dial offers the recount rule at its middle and quicker settings and the history rule at its cautious settings (interval level 95% and above), where it catches more losses for the same false alarms. The frozen S4 target stays on the recount rule at 90%. The choice is re-checked when the count error model is refitted after Phases 2 and 3, before the Phase 7 test run, and nothing is chosen on the test farms.
Evidence / reasoning: On the 4000 validation farms with the Phase 1 count error (`results/phase4/s4_validation_4000farms.json`), the history rule at 95% gives 0.89 false alarms per 100 surveys and 53% (46%-59%) of losses of 5 or more caught, against 0.61 false alarms per 100 surveys and 41% (34%-47%) for the recount rule at 95% and 1.45 false alarms per 100 surveys and 58% (51%-64%) at 90%. Near 1.5 false alarms the two rules are alike (H18 not held). With the S3 target's count error the history rule at 90% meets the S4 target (1.49 false alarms per 100 surveys and 73% (66%-78%)). It is harder to explain to a farmer why it fired, so its messages say how many checks since the records were last confirmed point the same way. Simulation results, not field results.
Easy option rejected (if any) and why: Replacing the recount rule everywhere, which the results do not support at the middle of the dial.
Status: adopted 2026-11-22 (owner)

## D-40: Run Phase 5 alongside Phases 2 and 3, and declare S5 only with the dashboard in the loop  (2026-11-22)
Options considered: wait for Phases 2 and 3 before Phase 5; run Phase 5 now on the laptop's CPU while Phase 2 trains on rented GPUs; end S5 at the result record the dashboard reads.
Choice: Phase 5 runs now, because its mission, flight, capture and hand-off work does not depend on the detector's final accuracy or on the Phase 3 matcher; both enter through fixed interfaces (the detection error model, refitted after Phase 2, and one duplicate-removal interface, which holds the Phase 1 method until Phase 3 replaces it). S5 keeps the brief's definition (plan, fly, capture, count at home base, dashboard): it is declared passed only when the CI job also sends the record to the Phase 6 dashboard backend and checks what is shown, and it is re-run and re-declared after the Phase 3 matcher replaces the baseline. Until then progress is reported as "S5 up to the result record". Every record states the matcher and error-model versions it used.
Evidence / reasoning: Brief §12 allows a phase to start early when the work is clearly independent; the parts that are not independent are isolated behind interfaces and re-run when they change. The independent review of the Phase 5 plan found that ending S5 at the record would quietly redefine the specification.
Easy option rejected (if any) and why: Declaring S5 passed at the end of Phase 5 without the dashboard, which would be an easier test than the one specified.
Status: adopted 2026-11-22 (owner)


## D-42: Freeze hashes independent of line endings; re-freeze after the S3 truth grew to 20 flights  (2026-11-22)
Options considered: keep hashing the raw bytes of each frozen file; hash text files with their line ends normalised to LF (binary files unchanged); re-freeze only the manifests that fail.
Choice: Text files are hashed with LF line ends (`herdsight.utils.freeze.sha256`), and all four manifests (phase1_splits, phase1_specs, phase2_prereg, phase3_prereg) are re-frozen with that hash. The earlier manifests are kept in `results/frozen/history/` with the date 2026-11-22.
Evidence / reasoning: The repository stores text with LF line ends (`.gitattributes`), but the Windows working copy holds several frozen files with CRLF, so every earlier manifest matched only that one working copy and would have failed on a clean checkout (for example in CI). Before re-freezing, every file named in the earlier manifests was checked against its recorded raw hash: all matched except `results/data/s3_truth.json`. That file changed on purpose when the owner verified the 10 remaining flights (review round 3, 2026-11-22; the first 10 flights' counts are unchanged), which also changes the Phase 1 specification freeze. The Phase 3 pre-registration had been frozen from a copy of the same content with CRLF line ends. After re-freezing, each manifest was checked against the committed copies in git, and all matched. No setting, target or count changed in the re-freeze.
Easy option rejected (if any) and why: Re-freezing only the failing manifests on this computer would have hidden the problem until a run elsewhere.
Status: adopted 2026-11-22

## D-41: Fly the survey with offboard setpoints, by the rule fixed before the comparison  (2026-11-22)
Options considered: a PX4 custom mode through `px4-ros2-interface-lib`; raw offboard setpoints, with PX4 set to return if they stop; an uploaded PX4 mission with distance camera triggering.
Choice: Offboard setpoints (`offboard_mission_node`, now the default in `sim/bridge/run_flight.sh`). The custom-mode mission stays in the repository for comparison.
Evidence / reasoning: The rule was written in the Phase 5 plan before the first comparison flight. The first six flights exposed two faults in the test harness (the height check counted the landing, and the offboard controller kept streaming during the return), so all six were flown again with the rule unchanged (`results/phase5/d41_interfaces.json`). Custom mode and offboard both pass: 44 of 44 stills within 1.5 m of their planned points, and after the companion software was lost at still 22, PX4 took over in 1.15 s and 1.05 s, stayed above 82 m until back over launch, and landed within 0.05 m of it. The mission fails: it took 40 of 44 stills (the last point of each line, an effect of how the test stops the trigger) and cannot insert herd-mode targets without a re-upload. Offboard flew the survey in 150.5 s and custom mode in 170.2 s, so custom mode is 13% slower, outside the 5% allowance that would have kept it. Limits: one flight per case on one simulated world; the time difference may shrink if PX4's trajectory settings for the custom mode are tuned, which was not tried; offboard's safe return depends on `COM_OBL_RC_ACT` being set to return; inserting herd-mode targets is a property of the interface and was not flown in this comparison. Simulation results, not field results.
Easy option rejected (if any) and why: Keeping the custom mode because it was already built and was the plan's expectation, which would have ignored the rule written before the flights.
Status: adopted 2026-11-22, by the fixed rule; the offboard mission has since flown a normal survey, herd mode and a node crash (`results/phase5/offboard_checks.json`) and link loss; GNSS loss and the geofence margin remain open safety limits (`results/phase5/`, Phase 5 plan)

## D-43: An ideal stabilised camera for the simulated bounding-box sensor  (2026-11-23)
Options considered: PX4's simulated gimbal (x500_gimbal); a camera fixed to the drone's body; a separate camera model with no collision or gravity, kept over the drone and pointing straight down.
Choice: The separate camera model (`sim/bridge/add_survey_camera.py`, `sim/bridge/camera_bridge.py`), switched on with `BBOX=1` in `sim/bridge/run_flight.sh`. The drone model and its PX4 airframe are unchanged.
Evidence / reasoning: PX4's simulated gimbal did not move when commanded, through PX4 or directly on its joint, in the farm world or PX4's default world. A body-fixed camera placed animals about 13 to 15 m off, in opposite directions on opposite passes, consistent with the drone leaning about 10 degrees at 8 m/s. The separate model stays within 0.25 m of the drone at every still. A new pose command is sent only after the last has taken effect, because a backlog left the camera about 70 s behind. A frame shows the camera 0.1 s before its time stamp, measured on a separate calibration flight of the same world by lining up animals seen on opposite passes.
Easy option rejected (if any) and why: Keeping the body-fixed camera and accepting the shift, which would have broken duplicate removal between passes.
Status: adopted 2026-11-23. The simulated results therefore exclude camera pointing error from a real gimbal; this is a stated limitation of every count from these flights.

## D-44: First new-farm experiment: the G2 recipe under leave-one-farm-out, without hard negatives  (2026-11-23)
Options considered: pseudo-labelling unlabelled images; ground-sample-distance normalisation (G5); the existing G2 appearance and crowding recipe, measured on the farm it never saw; the same recipe with the G3 hard negatives.
Choice: The G2 recipe under leave-one-farm-out, 72 epochs per held-out farm, without hard negatives (`gpu/runpod/job_g6g2.sh`). It is compared with the reference recipe's leave-one-farm-out result (`results/phase2/g6_lofo.json`).
Evidence / reasoning: G6 showed that the largest accuracy loss is on an unseen farm (recall 0.553 against 0.728). G2 was built for changes in appearance but was only scored under the official cross-validation, where most farms are partly seen, so its effect on a new farm is unmeasured. The G3 hard negatives were mined by models trained on farms that would now be held out, so using them here would leak information about the held-out farm.
Easy option rejected (if any) and why: Reusing the hard negatives to match G2 exactly, which would make the unseen-farm result look better than it is.
Status: done 2026-11-24. The G2 recipe did not replace the reference under leave-one-farm-out (D-49).

## D-45: Use each animal's true logged position when counting Gazebo flights  (2026-11-23)
Options considered: keep working positions back from the bounding-box pixels and the camera pose; log the camera pose more often; retune the frame delay; use each animal's true logged position, keeping the sensor's record of which animals were visible and the measured detection, position and geotag error models.
Choice: The true logged position at the trigger (`python/herdsight/sim/home_base.py`). Visibility still comes from the Gazebo sensor.
Evidence / reasoning: The simulated camera moves in steps of up to about 0.5 m, so the same animal landed about 0.8 m apart between photos against a matching limit of about 1 m, and was counted twice. Logging the pose five times as often did not remove this, and the best frame delay still left many split animals. Real position error is already added by the error models, so the camera's stepping was an extra error of the test harness.
Easy option rejected (if any) and why: Tuning the frame delay on the flights being checked, which would fit the harness to the answer.
Status: adopted 2026-11-23. Gazebo counts before this date are superseded. A flight in which part of the herd walks about 105 m still overcounts (median 50 for 30 animals), because moving animals are seen twice; this is a real limitation and is not corrected.

## D-46: Take off at least the clearance radius outside the camp fence  (2026-11-23)
Options considered: check animal positions before take-off; ask the pilot to confirm the area is clear; require a launch point at least the clearance radius (30 m) outside the fence.
Choice: The launch point must be at least 30 m from the camp fence (`python/herdsight/mission/launch.py`). The simulated world now places the fence 35 m from take-off, and `sim/bridge/make_world.py` refuses a closer launch.
Evidence / reasoning: With the fence 20 m from take-off, one simulated flight climbed with an animal 27.7 m away, breaking the rule that the drone only climbs or descends with no animal within 30 m. Animals move, so a check at take-off would not hold for the landing. The same layout re-flown with the new rule passed; the nearest animal to the take-off point was 41 m at the start and 49 m at the end.
Easy option rejected (if any) and why: Reducing the clearance radius so the flight passes, which would weaken the welfare limit to fit the result.
Status: adopted 2026-11-23. The rule relies on the fence holding the animals. Other recorded flights used the earlier 20 m layout and passed the welfare check; the geofence and GPS-loss measurements also used it and need repeating on the new layout.

## D-47: Trigger the geofence return 5 m before the 150 m fence  (2026-11-23)
Options considered: raise the 165 m limit; use PX4's experimental predictive geofence; trigger the return at 145 m, keeping the 165 m limit unchanged.
Choice: Set `GF_MAX_HOR_DIST` to 145 m.
Evidence / reasoning: With the trigger at 150 m the drone reached 165.37 m on the current layout, 0.37 m past the limit (`results/phase5/geofence_check_new_layout.json`). With the trigger at 145 m it reached 160.11 m and 160.18 m on two farm layouts and returned safely (`results/phase5/geofence_check_trigger_145m.json`, `geofence_check_trigger_145m_layout2.json`).
Easy option rejected (if any) and why: Raising the limit to 166 m, which would change the test to fit the result.
Status: adopted 2026-11-24, by the rule written before the wind flights. With a steady wind of 5 m/s and 10 m/s blowing the way the drone crosses the trigger, and 10 m/s blowing straight outward, the drone reached at most 159.6, 159.8 and 159.0 m, stayed at least 80.0 m up until back over launch and landed within 0.07 m of launch (`results/phase5/geofence_wind_check.json`). The wind acted: the drone leaned 3.5, 7.0 and 6.9 degrees while holding for a still, against 0.27 with no wind. The position controller absorbs a steady wind, so the overshoot comes from braking at 8 m/s, as without wind. Limits: one flight per case on one world, a steady wind without gusts, and an assumed drag area (`sim/bridge/add_wind.py`); a gust test remains for real flights.

## D-48: Upright detector counts; the angle is added per counted animal  (2026-11-24)
Options considered: ship RTMDet-R as the oriented detector; try a second oriented family (detectron2 rotated Faster R-CNN); keep upright D-FINE for counting and predict an angle for each counted animal.
Choice: Keep upright D-FINE for counting and detection; give each counted animal an angle from a separate step. This supersedes D-8's oriented boxes as the primary output.
Evidence / reasoning: G7's rule, written before the run, says that if the best oriented model is worse than the best upright model beyond the seed spread, the upright detector counts. RTMDet-R's centre AP was 0.218 against 0.545 (-0.326, interval -0.383 to -0.279), far beyond any seed spread, and its count error was higher (`results/phase2/g7_comparison.json`). Oriented boxes remain an output requirement for herd-mode framing and later identification.
Easy option rejected (if any) and why: Retuning RTMDet-R until it comes closer, which the rule forbids and which would use the remaining GPU budget on the weaker family.
Status: adopted 2026-11-24 (owner). A second oriented family was allowed by the plan because the first clearly underperformed; it is not being tried, and the remaining GPU budget goes to the new-farm work instead.

## D-49: The G2 recipe does not replace the reference on unseen farms; G5 is the next new-farm experiment  (2026-11-24)
Options considered: adopt the G2 recipe because it found more cattle on Jalogny; keep the reference recipe; repeat the G2 leave-one-farm-out runs with more seeds before deciding.
Choice: Keep the reference recipe as the current best. The next new-farm experiment is resolution normalisation (G5), as planned for this outcome.
Evidence / reasoning: Both columns are leave-one-farm-out, so each farm is predicted by a model that never saw it and each threshold is set on the other two farms (`results/phase2/g6g2_vs_g6ref_lofo.json`, 1,323 images). Against the reference, the G2 recipe shows no difference in centre AP (-0.009, interval -0.048 to 0.028), a higher count error (+0.33 animals per image, interval 0.08 to 0.80) and more false animals on empty images (+0.28, interval 0.04 to 0.78). It found more cattle on Jalogny (recall 0.77 against 0.72) but fewer on Derval and Mauron. The selection rule in `configs/phase2.yaml` requires a candidate to be better on centre AP and count error and not to lose under leave-one-farm-out; G2 meets neither, and the count-error loss is outside the close-call zone, so repeats are not needed. Under the official cross-validation G2 had already tied the reference (`results/phase2/g2_comparison.json`). Jalogny and Mauron were predicted on a rented A40 (session g6g2-score-2); the predictions agree with a partial home-base CPU run within one animal on 10 of 368 images.
Easy option rejected (if any) and why: Adopting G2 for its Jalogny gain, which would pick the one farm that favours it and ignore the higher count error that the rule weighs.
Status: adopted 2026-11-24, by the fixed rule.

## D-50: Dark-coat, tight-group and outline training (G12) does not replace the reference on unseen farms; seed variation is larger than the resampling intervals  (2026-11-25)
Options considered: adopt the G12 recipe (coats darkened inside each animal's outline, exact animal shapes pasted into tight groups, and darker scenes); keep the reference recipe; train more seeds of both before deciding.
Choice: Keep the reference recipe. The onboard detector (G10) is trained with the reference recipe. A second G12 seed was scored to complete the record (below); the third reference seed was lost (its session overran and its results were never uploaded) and is not repeated, since two seeds already give the noise yardstick.
Evidence / reasoning: All columns are leave-one-farm-out on the same 1,323 images. Against the reference trained with seed 0, G12 shows no difference in centre AP (-0.002, interval -0.015 to 0.009), recall (+0.017, interval -0.024 to 0.049) or count error (+0.13 animals per image, interval -0.04 to 0.37), and more false animals on empty images (+0.17, interval 0.06 to 0.40) (`results/phase2/g12_vs_g6ref_lofo.json`). Against the same reference recipe trained with seed 1, every difference is close to zero (count error -0.02, interval -0.17 to 0.19; empty-image false animals +0.02, interval -0.13 to 0.15) (`results/phase2/g12_vs_g6ref_s1_lofo.json`). The reference recipe trained with seed 1 instead of seed 0 itself differs by +0.15 animals per image in count error (interval 0.04 to 0.35) and +0.15 false animals on empty images (interval 0.04 to 0.40) (`results/phase2/g6ref_s1_vs_g6ref_lofo.json`): the intervals resample images and survey days, not training runs, so a single pair of runs can look different by chance. G12's differences are within this seed-to-seed spread, and it found fewer cattle on Derval (recall 0.18 against 0.21 and 0.25). The selection rule in `configs/phase2.yaml` requires a gain on centre AP and count error; G12 shows neither.
Easy option rejected (if any) and why: Adopting G12 for its higher Jalogny recall (0.77 against 0.72 and 0.76), which picks the farm that favours it. Also rejected: treating the seed-1 comparison alone as a tie that allows adoption, since the rule asks for a gain, not a tie.
Status: adopted 2026-11-25, by the fixed rule. Consequence for D-49: G2's count-error loss (+0.33) is about twice the seed-to-seed difference, so that decision stands, but future recipe comparisons are judged against more than one reference seed.
Record (2026-11-25): G12 trained with seed 1 agrees. Against the reference seed 0: centre AP -0.013 (interval -0.029 to 0.004), count error +0.09 (interval -0.04 to 0.28), empty-image false animals +0.10 (interval 0.02 to 0.28) (`results/phase2/g12_s1_vs_g6ref_lofo.json`). Against the reference seed 1: centre AP -0.017 (interval -0.029 to -0.008), count error -0.06 (interval -0.25 to 0.13) (`results/phase2/g12_s1_vs_g6ref_s1_lofo.json`). No gain on either measure the rule requires.
