"""Error models for simulated flights (python/herdsight/sim/errors.py)."""

import json

import numpy as np

from herdsight.sim.errors import DetectionModel, GeotagModel, RegistrationModel
from herdsight.utils.config import REPO_ROOT


def test_geotag_model_reproduces_the_real_disagreement():
    real = json.loads((REPO_ROOT / "results" / "data" / "geolocation_check.json").read_text())
    real = real["conventions"][real["measured_convention"]]
    g = GeotagModel.load()
    rng = np.random.default_rng(0)
    d = []
    for _ in range(400):  # flights
        e = g.errors(12, g.flight_sigma(rng), rng)
        d += np.linalg.norm(e[1:] - e[:-1], axis=1).tolist()
    assert abs(np.median(d) - real["median_m"]) / real["median_m"] < 0.35
    assert abs(np.quantile(d, 0.9) - real["p90_m"]) / real["p90_m"] < 0.35


def test_detection_model_clusters_misses_and_favours_large_lone_animals():
    m = DetectionModel.load()
    rng = np.random.default_rng(1)
    n = 4000
    big = m.found(np.full(n, 90.0), np.zeros(n), np.ones(n), 0.0, rng).mean()
    small = m.found(np.full(n, 30.0), np.zeros(n), np.ones(n), 0.0, rng).mean()
    crowded = m.found(np.full(n, 90.0), np.ones(n), np.ones(n), 0.0, rng).mean()
    hidden = m.found(np.full(n, 90.0), np.zeros(n), np.full(n, 0.3), 0.0, rng).mean()
    assert big > small and big > crowded and big > hidden
    new_farm = np.mean(
        [
            m.found(np.full(50, 60.0), np.zeros(50), np.ones(50), m.new_farm_effect, rng).mean()
            for _ in range(40)
        ]
    )
    assert new_farm < 0.4  # a farm like Derval
    per_flight = [
        m.found(np.full(50, 60.0), np.zeros(50), np.ones(50), m.flight_effect(rng), rng).mean()
        for _ in range(200)
    ]
    assert np.std(per_flight) > 0.1  # misses cluster by flight, not independently


def test_registration_model_matches_real_acceptance():
    r = RegistrationModel.load()
    rng = np.random.default_rng(2)
    out = [r.outcome(rng) for _ in range(5000)]
    accepted = np.mean([o is not None for o in out])
    assert abs(accepted - r.accepted_share) < 0.02 and 0.85 < accepted < 0.97
