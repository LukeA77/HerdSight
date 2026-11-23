"""Certain pass/fail checks on one finished Gazebo flight (Phase 5 plan, section 8).

These are the per-flight conditions for S5 up to the result record: the planned photos were taken
(or the flight is correctly reported as ended early), the drone landed, the welfare limit held, the
hand-off happened, and the record is valid, for the right camp, marked simulated, with coverage
filled and plain wording. An early return must give status `unknown` and no alert.

No check asks whether the count is right or its range contains the truth: a 90% range misses about
one flight in ten by design, so that belongs in a many-seed run. Simulation only.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator, FormatChecker

from herdsight.herd import messages
from herdsight.utils.config import load_config


def closest_approach(
    frames: dict[str, list[dict[str, Any]]], radius_m: float
) -> dict[str, float | int | None]:
    """Lowest drone height while within `radius_m` of any animal, from the logged true poses."""
    pose, herd = frames["pose"], frames["herd"]
    if not pose or not herd:
        return {"samples": 0, "lowest_height_within_radius_m": None}
    t_herd = np.array([r["sim_s"] for r in herd])
    lowest: float | None = None
    for row in pose:
        k = int(np.abs(t_herd - row["sim_s"]).argmin())
        animals = np.asarray(herd[k]["xy"], float).reshape(-1, 2)
        x, y, z = row["drone"]
        if len(animals) and np.min(np.hypot(animals[:, 0] - x, animals[:, 1] - y)) <= radius_m:
            lowest = float(z) if lowest is None else min(lowest, float(z))
    return {"samples": len(pose), "lowest_height_within_radius_m": lowest}


def check_flight(
    result: dict[str, Any], frames: dict[str, list[dict[str, Any]]], case: str
) -> dict[str, Any]:
    """`result` from `count_run`; `case` is `complete` or `early`."""
    if case not in ("complete", "early"):
        raise ValueError(f"unknown case {case!r}")
    cfg = load_config("mission")
    floor = float(cfg.require("herd_mode_height_m"))
    radius = float(cfg.require("animal_clearance_radius_m"))
    bundle, record = result["bundle"], result["record"]
    schema = json.loads(messages.SCHEMA.read_text(encoding="utf-8"))
    errors = list(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(record))
    text = " ".join([*record["notes"], *(a["message"] for a in record["alerts"])])
    approach = closest_approach(frames, radius)
    lowest = approach["lowest_height_within_radius_m"]
    checks = {
        "landed": bool(bundle["landed"]),
        "hand_off_has_every_still": bundle["photos_taken"] + len(bundle["photos_missing"])
        == bundle["photos_planned"],
        "welfare_poses_logged": bool(approach["samples"]),
        "welfare_height_held": bool(approach["samples"]) and (lowest is None or lowest >= floor),
        "record_valid": not errors,
        "record_right_camp": record["camp_name"] == bundle["camp"],
        "record_marked_simulated": record["simulated"] is True,
        "coverage_filled": record.get("coverage") is not None,
        "plain_wording": not messages.banned_words(text),
    }
    if case == "complete":
        checks["all_planned_photos_taken"] = bundle["outcome"] == "completed"
    else:
        checks["reported_as_ended_early"] = bundle["outcome"] == "ended_early"
        checks["status_unknown"] = record["status"] == "unknown"
        checks["no_alert"] = record["alerts"] == []
    return {
        "case": case,
        "passes": all(checks.values()),
        "checks": checks,
        "welfare_floor_m": floor,
        "clearance_radius_m": radius,
        "lowest_height_within_clearance_m": lowest,
    }
