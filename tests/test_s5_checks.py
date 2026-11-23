"""Certain per-flight S5 checks (python/herdsight/sim/s5_checks.py)."""

import copy

import pytest

from herdsight.herd import messages
from herdsight.herd.alerts import Rule
from herdsight.herd.sim import Survey
from herdsight.sim.s5_checks import check_flight, closest_approach
from herdsight.utils.config import load_config

FLOOR = float(load_config("mission").require("herd_mode_height_m"))
RADIUS = float(load_config("mission").require("animal_clearance_radius_m"))


def _result(coverage=1.0, outcome="completed", taken=4, missing=()):
    s = Survey(0.0, "Camp", 30, 30, {0.9: 27.0}, {0.9: 33.0}, False)
    record = messages.survey_record(
        s, farm_id="sim", expected=30, rule=Rule("recount", level=0.9), coverage=coverage
    )
    bundle = {
        "camp": record["camp_name"],
        "landed": True,
        "outcome": outcome,
        "photos_planned": taken + len(missing),
        "photos_taken": taken,
        "photos_missing": list(missing),
    }
    return {"bundle": bundle, "record": record}


def _frames(height=80.0, distance=100.0):
    return {
        "pose": [{"sim_s": 1.0, "drone": [0.0, 0.0, height], "camera": [0.0, 0.0, height]}],
        "herd": [{"sim_s": 1.0, "xy": [[distance, 0.0]]}],
    }


def test_complete_flight_passes():
    out = check_flight(_result(), _frames(), "complete")
    assert out["passes"] and out["lowest_height_within_clearance_m"] is None


def test_low_near_animals_fails_and_low_away_from_them_does_not():
    near = check_flight(_result(), _frames(FLOOR - 1, RADIUS - 1), "complete")
    assert not near["passes"] and not near["checks"]["welfare_height_held"]
    assert check_flight(_result(), _frames(FLOOR - 1, RADIUS + 1), "complete")["passes"]


def test_a_flight_without_logged_poses_fails():
    out = check_flight(_result(), {"pose": [], "herd": []}, "complete")
    assert not out["passes"] and not out["checks"]["welfare_poses_logged"]


def test_early_return_must_be_unknown_with_no_alert():
    early = _result(coverage=0.4, outcome="ended_early", taken=2, missing=(2, 3))
    assert check_flight(early, _frames(), "early")["passes"]
    wrong = copy.deepcopy(early)
    wrong["record"]["status"] = "ok"
    assert not check_flight(wrong, _frames(), "early")["checks"]["status_unknown"]


def test_missing_photos_fail_a_complete_flight():
    out = check_flight(_result(outcome="ended_early", taken=3, missing=(3,)), _frames(), "complete")
    assert not out["checks"]["all_planned_photos_taken"]


def test_unknown_case_is_refused():
    with pytest.raises(ValueError):
        check_flight(_result(), _frames(), "other")


def test_closest_approach_uses_the_nearest_herd_sample():
    frames = {
        "pose": [{"sim_s": 10.0, "drone": [0.0, 0.0, 50.0], "camera": [0.0, 0.0, 50.0]}],
        "herd": [{"sim_s": 0.0, "xy": [[0.0, 0.0]]}, {"sim_s": 10.0, "xy": [[500.0, 0.0]]}],
    }
    assert closest_approach(frames, RADIUS)["lowest_height_within_radius_m"] is None
