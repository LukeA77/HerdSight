"""Data hand-off after a simulated flight: the bundle home base starts from (Phase 5, step 7).

`make_bundle` reads one finished run of sim/bridge/run_flight.sh (plan.yaml, stills.csv,
herd_state.pkl, mission.log, px4.log) and records what actually happened:

* which planned photos were taken, each with PX4's position estimate at the trigger;
* herd-mode stills, told apart by being well below the survey height;
* coverage: the share of the camp inside the footprint of a survey still actually taken;
* the outcome: `completed` (every planned photo), `ended_early` (the drone landed with photos
  missing, wherever it came down) or `not_landed`;
* whether the coverage is enough for a missing-animal alert (configs/mission.yaml).

The bundle holds no animal count: the bounding-box camera that supplies detections to home base
is a later step, so nothing here can be read as a counting result. Simulation only.
"""

from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from herdsight.sim.herds import Camp
from herdsight.utils.config import load_config

HERD_STILL_HEIGHT_SHARE = 0.8  # a still below this share of the survey height is a herd still
GRID_STEP_M = 5.0


def read_stills(path: Path) -> list[dict[str, Any]]:
    """Stills from `ros2 topic echo --csv herdsight/still`; other lines are skipped."""
    out = []
    with path.open(encoding="utf-8", errors="replace", newline="") as fh:
        for r in csv.reader(fh):
            if len(r) > 8 and r[3].isdigit():
                out.append(
                    {
                        "t_s": int(r[0]) + int(r[1]) * 1e-9,
                        "photo_id": int(r[3]),
                        "enu_m": [float(r[4]), float(r[5]), float(r[6])],
                    }
                )
    return out


def camp_coverage(
    camp: Camp,
    centres: NDArray[np.float64],
    heading_rad: float,
    footprint_m: tuple[float, float],
    step: float = GRID_STEP_M,
) -> float:
    """Share of the camp inside at least one footprint (footprint is across, along; metres)."""
    lo, hi = camp.polygon.min(0), camp.polygon.max(0)
    xx, yy = np.meshgrid(np.arange(lo[0], hi[0], step), np.arange(lo[1], hi[1], step))
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    pts = pts[camp.contains(pts)]
    if not len(pts) or not len(centres):
        return 0.0
    c, s = np.cos(heading_rad), np.sin(heading_rad)
    rot = np.array([[c, -s], [s, c]])
    across, along = footprint_m
    seen = np.zeros(len(pts), bool)
    for centre in centres:
        d = (pts - centre) @ rot  # columns: along, across
        seen |= (np.abs(d[:, 0]) <= along / 2) & (np.abs(d[:, 1]) <= across / 2)
    return float(seen.mean())


def make_bundle(run: Path) -> dict[str, Any]:
    plan = yaml.safe_load((run / "plan.yaml").read_text(encoding="utf-8"))
    with (run / "herd_state.pkl").open("rb") as fh:  # written by sim/bridge/make_world.py
        camp = pickle.load(fh).camp
    planned = [p["id"] for s in plan["sorties"] for p in s["photos"]]
    height = float(plan["height_agl_m"])
    stills = read_stills(run / "stills.csv")
    survey = [s for s in stills if s["enu_m"][2] >= HERD_STILL_HEIGHT_SHARE * height]
    herd = [s for s in stills if s["enu_m"][2] < HERD_STILL_HEIGHT_SHARE * height]
    taken = sorted({s["photo_id"] for s in survey})
    missing = sorted(set(planned) - set(taken))
    centres = np.array([s["enu_m"][:2] for s in survey], float).reshape(-1, 2)
    coverage = camp_coverage(
        camp, centres, float(plan["heading_enu_rad"]), tuple(plan["footprint_m"])
    )
    px4_log = (run / "px4.log").read_text(encoding="utf-8", errors="replace")
    landed = "Disarmed by landing" in px4_log
    outcome = "not_landed" if not landed else ("ended_early" if missing else "completed")
    minimum = float(load_config("mission").require("min_coverage_for_alert"))
    return {
        "bundle_version": "1",
        "simulated": True,
        "camp": plan["camp"],
        "outcome": outcome,
        "landed": landed,
        "photos_planned": len(planned),
        "photos_taken": len(taken),
        "photos_missing": missing,
        "herd_mode_stills": len(herd),
        "coverage": coverage,
        "min_coverage_for_alert": minimum,
        "coverage_enough_for_alert": coverage >= minimum,
        "stills": survey + herd,
    }
