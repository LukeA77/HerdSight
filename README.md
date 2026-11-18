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
| `scripts/` | Freezing settings, one-shot test runs, filling docs from results |
| `results/` | Machine-written results (never edited by hand) |
| `tests/` | Automated tests |

Datasets and model weights are not stored in this repository. Instructions for fetching them will be in `docs/data.md`.

## Licence

Copyright (c) 2026 Luke Andrew. All rights reserved. See `LICENSE`.
