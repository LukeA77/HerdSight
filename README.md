# HerdSight

HerdSight uses a drone to count cattle in each camp (paddock), checks the count against the farm's records, and tells the farmer when animals may be missing. The aim is to help fight stock theft.

> **Status: early development.** Nothing here has been flown on a real farm. All work so far uses public datasets and simulation.

## What it will do

- Count cattle per camp from overlapping drone photos, without counting the same animal twice.
- Give every count a range, e.g. "142 cows (between 140 and 144)".
- Raise an alert only when the whole range is below the expected number, and keep false alarms rare.
- Show results on a phone-friendly dashboard written for farmers, not engineers.

## Repository layout

| Folder | What's in it |
|---|---|
| `docs/` | Decision log, hypotheses, data notes, open questions, limitations, and one plan per piece of work (`docs/plans/`) |
| `configs/` | Settings, each with a cited source |
| `core/` | C++ library: geometry, matching, count intervals (with Python bindings) |
| `python/herdsight/` | Python package: data loading, detection, counting, alerts, evaluation, simulation tools |
| `ros2/` | ROS 2 packages that fly the mission |
| `sim/` | Gazebo worlds, cattle models and the herd movement model |
| `dashboard/` | Farmer dashboard: `backend/`, `frontend/`, plain-language glossary |
| `scripts/` | Data inventory and splits, baselines, freezing settings, one-shot test runs, filling docs from results |
| `gpu/` | Training scripts that run on free cloud GPUs (Kaggle, Colab) |
| `results/` | Machine-written results (never edited by hand) |
| `tests/` | Automated tests |

Datasets and model weights are not stored in this repository. Instructions for fetching them will be in `docs/data.md`.

## Development

Python 3.12 or later.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt   # Windows: .venv\Scripts\python
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/pytest
```

The C++ core (`core/`) builds on Linux (Ubuntu 24.04, natively or in WSL) with CMake, Ninja, Eigen and GoogleTest:

```bash
sudo apt install build-essential cmake ninja-build libeigen3-dev libgtest-dev
scripts/build_core.sh            # sanitizer build + tests, then the Python module
HERDSIGHT_REQUIRE_CORE=1 pytest  # includes the C++/Python parity tests
```

Large files (datasets, weights, run outputs) live outside the repository, in the folder named by the `HERDSIGHT_STORAGE` environment variable.

## Licence

Copyright (c) 2026 Luke Andrew. All rights reserved. See `LICENSE`.
