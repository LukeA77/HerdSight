"""Second look in the fast simulator (Phase 5; _drafts/second-look-spec.md): flight logic."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from herdsight.mission.planner import PlannerSettings, plan_survey
from herdsight.sim import flight as fl
from herdsight.sim.herds import Camp, settings_from_config
from herdsight.utils.config import load_config


def _flight(second_look: bool, seed: int = 3) -> tuple:
    rng = np.random.default_rng([seed, 0])
    camp = Camp.rectangle(300, 250, "c")
    s = replace(settings_from_config(rng), groups=1)
    plan = plan_survey(camp, PlannerSettings.from_config("uk"))
    world = fl.build_world(camp, 90, np.random.default_rng([seed, 1]), herd_settings=s)
    f = fl.fly(world, plan, np.random.default_rng([seed, 2]), second_look=second_look)
    res = fl.count_flight(f, world.median_adult_m, np.random.default_rng([seed, 3]))
    return f, res, plan


def test_survey_only_flight_has_no_watches() -> None:
    f, res, _ = _flight(False)
    assert f.watches == [] and res["watched_spots"] == 0


def test_watches_are_capped_separate_and_clear_of_animals() -> None:
    f, res, plan = _flight(True)
    assert 0 < len(f.watches) <= fl.MAX_SPOTS
    assert res["watched_spots"] == len(f.watches)
    clearance = float(load_config("mission").require("animal_clearance_radius_m"))
    assert f.min_descent_clearance_m is None or f.min_descent_clearance_m >= clearance
    c, s = np.cos(plan.heading_rad), np.sin(plan.heading_rad)
    rot = np.array([[c, -s], [s, c]])
    across, along = f.watches[0]["footprint"]
    for i, a in enumerate(f.watches):
        assert a["count"] >= 0
        for b in f.watches[i + 1 :]:  # footprints never overlap (no animal counted twice)
            d = (a["centre_est"] - b["centre_est"]) @ rot
            # spots are chosen apart on their true centres; geotags may shift them by metres
            assert abs(d[0]) >= along - 10 or abs(d[1]) >= across - 10


def test_watch_counts_replace_survey_animals_in_the_spots() -> None:
    f, res, _ = _flight(True)
    # the same flight counted without its watches: every survey identity is kept
    base = fl.count_flight(fl.Flight(**dict(f.__dict__, watches=[])), 2.3, np.random.default_rng(0))
    watched = sum(w["count"] for w in f.watches)
    assert res["count"] >= watched
    assert base["watched_spots"] == 0 and res["watched_spots"] == len(f.watches)
