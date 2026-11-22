import numpy as np

from herdsight.count.evidence import Logistic, MotionModel
from herdsight.mission.planner import PlannerSettings, plan_survey
from herdsight.sim.flight import build_world, count_flight, fly
from herdsight.sim.herds import Camp
from herdsight.sim.phase3 import PositionModels, phase3_counts


def _models():
    # closer is more likely one animal; no other preference
    lo = Logistic(np.zeros(4), np.ones(4), np.array([3.0, -2.0, 0.0, 0.0, 0.0]))
    return PositionModels(MotionModel(0.5, 0.002), lo, lo)


def test_phase1_count_matches_the_simulator_and_all_methods_run():
    camp = Camp.rectangle(250, 180, "t")
    world = build_world(camp, 30, np.random.default_rng(1))
    plan = plan_survey(camp, PlannerSettings.from_config("uk"))
    flight = fly(world, plan, np.random.default_rng(2))
    ref = count_flight(flight, world.median_adult_m, np.random.default_rng(3))
    res = phase3_counts(flight, world.median_adult_m, np.random.default_rng(3), _models(), 120)
    assert res["phase1"] == ref["count"]  # same draws, same Phase 1 count
    assert res["detections"] == ref["detections"]
    for k in ("greedy_fixed", "global_fixed", "greedy_movement", "global_movement"):
        assert 0 < res[k] <= res["detections"]
