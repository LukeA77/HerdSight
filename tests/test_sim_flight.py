"""Survey flight and home-base count without ROS (python/herdsight/sim/flight.py)."""

import json

import numpy as np
import pytest
from jsonschema import Draft202012Validator, FormatChecker

from herdsight.herd import messages
from herdsight.mission.planner import PlannerSettings, plan_survey
from herdsight.sim.flight import build_world, count_flight, fly, run_survey
from herdsight.sim.herds import Camp

VALIDATOR = Draft202012Validator(
    json.loads(messages.SCHEMA.read_text(encoding="utf-8")), format_checker=FormatChecker()
)
CAMP = Camp.rectangle(300, 200, "River camp")
PLAN = plan_survey(CAMP, PlannerSettings.from_config("uk"))


@pytest.fixture(scope="module")
def full():
    rng = np.random.default_rng(11)
    world = build_world(CAMP, 40, rng)
    return run_survey(world, PLAN, rng, expected=40)


def test_full_survey_gives_a_valid_plain_record(full):
    rec = full["record"]
    assert not list(VALIDATOR.iter_errors(rec))
    assert rec["simulated"] is True and rec["camp_name"] == "River camp"
    assert rec["coverage"] == 1.0 and len(full["flight"].stills) == PLAN.n_photos
    text = " ".join(rec["notes"] + [a["message"] for a in rec["alerts"]])
    assert not messages.banned_words(text)
    assert "matcher" in rec["pipeline"]["version"] and "detection" in rec["pipeline"]["version"]


def test_an_early_return_never_raises_a_missing_alert():
    rng = np.random.default_rng(12)
    world = build_world(CAMP, 40, rng)
    out = run_survey(world, PLAN, rng, expected=40, stop_after=PLAN.n_photos // 3)
    rec = out["record"]
    assert not list(VALIDATOR.iter_errors(rec))
    assert rec["coverage"] < 0.9 and rec["status"] == "unknown" and rec["alerts"] == []
    assert len(out["flight"].stills) == PLAN.n_photos // 3


def test_the_count_depends_on_what_was_captured():
    rng = np.random.default_rng(13)
    world = build_world(CAMP, 40, rng, calf_share=0.0, trees_per_ha=0.0)
    flight = fly(world, PLAN, rng)
    base = count_flight(flight, world.median_adult_m, np.random.default_rng(0))["count"]
    for s in flight.stills:  # the camera saw nothing
        s.points, s.detected = s.points[:0], s.detected[:0]
    assert count_flight(flight, world.median_adult_m, np.random.default_rng(0))["count"] == 0
    assert base > 0


def test_a_new_farm_like_derval_finds_fewer_animals():
    found, counts = {}, {}
    for new in (False, True):
        d, c = [], []
        for seed in range(4):
            rng = np.random.default_rng(100 + seed)
            world = build_world(CAMP, 40, rng, calf_share=0.0, trees_per_ha=0.0)
            out = run_survey(world, PLAN, rng, expected=40, new_farm=new)
            d.append(out["count"]["detections"])
            c.append(out["count"]["count"])
        found[new], counts[new] = np.mean(d), np.mean(c)
    assert found[True] < 0.75 * found[False]
    assert counts[True] < counts[False]


def test_same_seed_same_flight():
    out = []
    for _ in range(2):
        rng = np.random.default_rng(14)
        world = build_world(CAMP, 30, rng)
        out.append(run_survey(world, PLAN, rng, expected=30)["count"])
    assert out[0] == out[1]


def test_herd_mode_respects_the_welfare_limits():
    from dataclasses import replace

    from herdsight.sim.herds import settings_from_config
    from herdsight.utils.config import load_config

    cfg = load_config("mission")
    rng = np.random.default_rng(21)
    s = replace(settings_from_config(rng), groups=1, grazing_share=0.0, walking_share=0.0)
    world = build_world(CAMP, 60, rng, herd_settings=s, trees_per_ha=0.0)
    flight = fly(world, PLAN, rng, herd_mode=True, force_trigger=True)
    herd_stills = [st for st in flight.stills if st.kind == "herd"]
    assert flight.herd_modes >= 1 and herd_stills
    focal = PLAN.height_agl / PLAN.gsd_m
    for st in herd_stills:
        assert st.gsd * focal >= cfg.require("herd_mode_height_m") - 1e-6  # never below 40 m
    assert flight.min_descent_clearance_m >= cfg.require("animal_clearance_radius_m")
    res = count_flight(flight, world.median_adult_m, np.random.default_rng(0))
    assert res["herd_mode_stills"] == len(herd_stills)
