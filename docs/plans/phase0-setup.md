# Phase 0: setup

## Goal

A repository that later phases can build on with confidence:

- a Python package (`herdsight`) with linting and tests;
- the reusable parts of drone-ranger brought in, with their tests passing;
- a C++ core (CMake, C++20, GoogleTest, pybind11) with one Python/C++ parity test, so the toolchain is proven before real algorithms move across;
- Continuous Integration (CI) on GitHub Actions that runs all of the above on every push;
- a single, checked way to find the external storage folder for datasets, weights and runs.

Exit criteria (from the brief): CI green; core library imported and tested.

## Approach

The work is split in two, because the C++ toolchain is not yet installed on the development machine.

**0a: Python (now)**

1. `pyproject.toml` at the repository root; package source in `python/herdsight/`. Python 3.12 (development) and 3.13 (current Debian/Raspberry Pi OS) both tested in CI.
2. Bring in the drone-ranger modules HerdSight needs (see D-1 below), with their tests. Changes are limited to import paths. Each file records where it came from (repository and commit).
3. `herdsight.utils.paths`: finds the storage folder from the `HERDSIGHT_STORAGE` environment variable. It fails with a clear message if the variable is missing or the drive is not connected, rather than quietly writing to the laptop's own disk.
4. `scripts/fill_docs.py` (fills `{{RESULT:...}}` tokens in documents from `results/`) and `scripts/freeze.py` (hashes frozen settings; the list of files per phase lives in `configs/freeze.yaml`, not in the code).
5. Ruff for linting, pytest for tests, and a pinned lock file for CI.
6. A GitHub Actions workflow with the Python job.

**0b: C++ (after the toolchain is installed in the Linux environment, WSL Ubuntu 24.04)**

1. `core/` CMake project: C++20, Eigen, GoogleTest; address and undefined-behaviour sanitizer builds.
2. One function ported from Python: rotation from yaw, pitch and roll (`R_wb_from_ypr`), which every ground projection uses. It's exposed to Python through pybind11 and checked against the Python reference to 1e-12 on random angles.
3. A C++ job added to CI (Ubuntu 24.04, with sanitizers).

### What is reused from drone-ranger, and what is not

| Taken | Why |
|---|---|
| `geometry/frames`, `camera`, `ranging`, `ground_cov`, `terrain`, `uncertainty` | Projecting detections to the ground with a position covariance (D-9 step 1) |
| `geometry/kalman` | Ground-plane constant-velocity filter; a starting point for the animal-motion model (D-9 step 2) |
| `evaluate/conformal`, `evaluate/stats` | Split conformal intervals and bootstrap intervals (§9.4, §7) |
| `utils/config`, `seeds`, `env_report` | Configs with cited sources and placeholders; reproducible seeds; environment record in each result |

| Not taken (now) | Why |
|---|---|
| Navigation filter (`navigation/`, MSCKF) | HerdSight's camera positions come from geotagged stills and the autopilot's own estimator (PX4 EKF2), not from a visual-inertial filter. Revisit if the logged position accuracy turns out to limit duplicate removal |
| Frame-to-frame tracker (`track/`) | Superseded by global matching across stills (D-9); a frame tracker causes the chain errors D-9 avoids |
| ONNX detector wrappers, tiling | Written for upright-box detectors at 640 px; HerdSight's detector is oriented boxes at full resolution. Rewritten in Phase 2 against the chosen model |
| Search, multi-drone communication | Specific to search and rescue. Revisit if multiple drones survey one farm |
| `geometry/imm` (multiple-model filter) | Not needed until the motion model is chosen in Phase 3 |

## Alternatives considered

- **A separate shared library repository**, versioned and pinned by both projects (the brief's first choice in D-1).
- **Depending on drone-ranger directly**, pinned to a commit.
- **Copying (vendoring) the needed modules** with provenance (chosen; see review).

## How success is measured

- All vendored tests pass unchanged apart from import paths. Any test that has to change is listed with the reason.
- New tests for `paths` and `freeze`.
- `ruff check` clean.
- CI green on Python 3.12 and 3.13 (0a), plus the C++ job with sanitizers and the parity test (0b).

## Risks

- Vendored tests may rely on drone-ranger fixtures or datasets (one known: `test_ground_cov` uses an AU-AIR helper as an independent reference).
- Floating-point differences between Windows (development) and Linux (CI).
- The free training environments (Kaggle, Colab) ship their own PyTorch, so a strict lock file for training would break them. The lock file covers development and CI only; training environments get a separate, short, pinned list (Phase 2).
- WSL Ubuntu currently lives on the laptop's own disk; the C++, ROS 2 and Gazebo toolchains add well over 10 GB.

## Review

**Red flags checked:**

- *Shrinking data or models:* not applicable in this phase.
- *Easier metric:* not applicable.
- *Test data related to training data:* not applicable.
- *Mocking something claimed as implemented:* the C++ core is a skeleton in 0b with one real ported function, and this is stated. No algorithm is claimed in C++ until it is ported and parity-tested.
- *Quietly narrowing scope:* not reusing the navigation filter could be a narrowing, so it is listed above with the condition for revisiting it.
- *Assuming without measuring:* the claim that vendored code behaves the same here is backed by running its own tests, not assumed.
- *Familiar rather than best:* pip with setuptools and a lock file was chosen partly because drone-ranger used it. uv gives faster installs and a cross-platform lock, which is better in principle. But Kaggle and Colab are driven by pip, and one tool everywhere avoids two dependency descriptions drifting apart. Decision: pip now. Revisit uv if dependency resolution becomes a problem.
- *C++ and Python maths without a parity test:* the parity test is part of 0b's exit criteria, not deferred.

**1. What is the best way, ignoring effort?** A shared, versioned core library with its own CI, used by both projects. This plan does not do that, for three reasons:

- drone-ranger is finished, and its published results were produced by frozen code. Refactoring it to depend on a new library would change code behind published numbers.
- A library with a single active user adds release overhead for no practical gain.
- Under D-14, the geometry HerdSight relies on moves to C++, so the Python copy becomes the reference implementation, not the production path.

The vendored modules keep their names and tests, and record their source commit, so extracting them later is mechanical. Logged as a refinement of D-1.

**2. What will most likely make this fail on a real farm?** Nothing in Phase 0 runs on a farm. The farm-relevant risk is storage paths: a field laptop without the external drive must fail loudly, not write data to the wrong place. `paths` handles this, with a test.

**3. How would a sceptical reviewer attack this?**

- "Copied code drifts from its source": provenance records the source commit. Drone-ranger is frozen, so there is nothing to drift from.
- "Tests were edited to pass": any change other than import paths is listed under Verification.
- "Works on your machine": CI runs on clean Linux images for both Python versions.

## Changes after review

- The navigation filter exclusion now states when to revisit it.
- A separate training dependency list was added to Risks, so Kaggle's preinstalled PyTorch is not fought by the lock file.
- The freeze file lists move from code into `configs/freeze.yaml`, so frozen phases are data, not code edits.

## Verification

### 0a: Python (2026-11-18)

- **Tests:** 183 passed, 0 skipped, 0 failed on Windows 11, Python 3.12.10. The Linux run on Python 3.12 and 3.13 happens in CI on the first push.
- **Lint:** `ruff check` and `ruff format --check` clean.
- **Vendored tests changed beyond import paths:**
  - `test_ground_cov.py`: the independent flat-ground reference (`ground_offset_oblique`, from drone-ranger's AU-AIR module) is copied into the test file, because the AU-AIR module itself is not reused. The test still compares against a separately written implementation.
  - `test_config.py`: three tests of drone-ranger's frame-rate budget (`latency_budget_ms`) were removed together with the function. That budget belongs to drone-ranger's onboard tracker; HerdSight's onboard budget (S7) is set after its baseline.
  - `test_env_report.py`: only the environment-report part of drone-ranger's `test_env_and_cli.py` (its command-line interface is not reused).
- **Vendored code changed beyond import paths:** `latency_budget_ms` removed from `utils/config.py`; `utils/env_report.py` lists HerdSight's packages instead of drone-ranger's.
- **Skipped tests resolved:** two frame tests skipped until `pymap3d` (an independent geodesy library, used only as a test reference) was added to the development dependencies. Two config tests skipped until the first config file (`configs/project.yaml`: animal class and operating countries) existed.
- **New tests:** `test_paths.py` (5: unset variable, missing drive, folder inside the repository, layout, unknown kind) and `test_freeze.py` (6).
- **Second review (did the build take an easier route than the plan?):** No. Every planned item for 0a is present, and nothing is stubbed. The run-once guard for test splits (§9.5) is not part of Phase 0; it is built in Phase 1 with the first test split.

### 0b: C++ (2026-11-18)

- **Toolchain:** Ubuntu 24.04 (WSL 2): g++ 13.3, CMake 3.28, Ninja 1.11, Eigen 3.4, GoogleTest 1.14; pybind11 3.1.0 from pip.
- **GoogleTest under address and undefined-behaviour sanitizers:** 5 of 5 passed. The tests cover identity at zero attitude, yaw 90° pointing east, positive pitch nose up, positive roll right wing down, and orthonormality plus the round trip on 1,000 random attitudes.
- **Parity with the Python reference** on 10,000 random attitudes: the largest difference is 1.1 × 10⁻¹⁶ for `R_wb_from_ypr` (one unit in the last place) and 0 for `ypr_from_R_wb`, against a test tolerance of 10⁻¹².
- **Full Python suite on Linux with the C++ module required** (`HERDSIGHT_REQUIRE_CORE=1`): 188 passed. On Windows, without a compiler, 183 passed and the parity test skipped, as designed.
- **Problem found and fixed:** with Ubuntu's packaged pybind11 (2.11), the parity test crashed (segmentation fault) the first time a NumPy array was passed into C++. pybind11 2.11 predates NumPy 2, and this project uses NumPy 2.5. The fix is pybind11 from pip, pinned in `requirements-lock.txt` (`>=2.12` in `pyproject.toml`); `scripts/build_core.sh` takes pybind11 from the Python environment. The crash shows why the parity test runs with real arrays rather than only scalars.
- **Build layout:** CMake builds outside the repository (`~/build/herdsight`). The module is written into `python/herdsight/` so `import herdsight._core` works without installing, and it is gitignored. Alternative considered: building the C++ through `pip install` (scikit-build-core). Rejected for now, because it would make a C++ compiler mandatory on every machine, including the Windows development laptop and Kaggle notebooks, which only need the Python reference. Revisit when the onboard nodes ship.
- **CI:** a `cpp-core` job runs the sanitizer build, builds the module, and runs the full Python suite with the core required.
- **Second review:** the build matches the plan; nothing is stubbed. One addition beyond the plan: `ypr_from_R_wb` (the inverse) was ported as well as `R_wb_from_ypr`, because a round-trip test needs both.

### Exit criteria

- CI green: confirmed on GitHub for 0a. 0b is confirmed when the `cpp-core` job passes on the next push.
- Core library imported and tested: yes. The Python modules have 183 tests; the C++ core has 5 GoogleTest tests plus 5 parity tests.
