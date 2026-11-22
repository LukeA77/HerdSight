"""Simulated herds (python/herdsight/sim/herds.py; Phase 5 plan, step 1)."""

from dataclasses import replace

import numpy as np

from herdsight.sim.herds import Camp, nn_mismatch_m, place_herd, settings_from_config
from herdsight.utils.config import load_config

BOUND = float(load_config("sim_worlds").require("nearest_neighbour_match_max_m"))


def _settings(seed=1, **kw):
    s = settings_from_config(np.random.default_rng(seed))
    return replace(s, **kw)


def test_camp_geometry():
    c = Camp.rectangle(100, 50)
    assert c.area_m2() == 5000
    assert c.contains([[10, 10], [150, 10], [-1, 5]]).tolist() == [True, False, False]
    tri = Camp(np.array([[0, 0], [10, 0], [0, 10]], float))
    assert tri.contains([[2, 2], [8, 8]]).tolist() == [True, False]


def test_placement_and_grazing_match_measured_spacing():
    camp = Camp.rectangle(400, 300)
    s = _settings(groups=2, grazing_share=1.0, walking_share=0.0, bout_s=1e9)
    rng = np.random.default_rng(2)
    herd = place_herd(camp, 150, s, rng)
    assert herd.n == 150 and camp.contains(herd.pos).all()
    assert nn_mismatch_m(herd.pos, s.nn_quantiles) <= BOUND
    for _ in range(600):  # 10 minutes of grazing
        herd.step(1.0, rng)
    assert nn_mismatch_m(herd.pos, s.nn_quantiles) <= BOUND
    nn = np.linalg.norm(herd.pos[:, None] - herd.pos[None], axis=2) + np.eye(herd.n) * 1e9
    assert nn.min() >= s.min_spacing * 0.9


def test_same_seed_same_run_and_fence_holds():
    camp = Camp.rectangle(120, 80)
    s = _settings(groups=3, walking_share=0.5, grazing_share=0.3, bout_s=120.0)
    runs = []
    for _ in range(2):
        rng = np.random.default_rng(7)
        herd = place_herd(camp, 60, s, rng)
        for _ in range(900):
            herd.step(1.0, rng)
            assert camp.contains(herd.pos).all()
        runs.append(herd.pos.copy())
    assert np.array_equal(runs[0], runs[1])


def test_animals_react_only_when_the_drone_is_low_enough():
    camp = Camp.rectangle(300, 300)
    for height, expect in ((0.0, False), (60.0, True)):
        s = _settings(groups=1, grazing_share=0.0, walking_share=0.0, reaction_height=height)
        rng = np.random.default_rng(3)
        herd = place_herd(camp, 40, s, rng)
        centre = herd.pos.mean(0)
        before = np.linalg.norm(herd.pos - centre, axis=1).mean()
        reacted = herd.step(1.0, rng, drone=(*centre, 50.0))
        after = np.linalg.norm(herd.pos - centre, axis=1).mean()
        assert reacted.any() == expect
        assert (after > before + 0.1) == expect
