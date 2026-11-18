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
