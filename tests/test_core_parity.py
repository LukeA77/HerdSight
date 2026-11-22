"""Parity: the C++ core (herdsight._core) must match the Python reference (herdsight.geo).

The C++ module is optional on machines without a compiler, so this test skips when it is not
built, unless HERDSIGHT_REQUIRE_CORE=1 is set. CI sets it, so there the check cannot be
skipped.
"""

import os

import numpy as np
import pytest

from herdsight.geo import frames as fr

if os.environ.get("HERDSIGHT_REQUIRE_CORE") == "1":
    from herdsight import _core
else:
    _core = pytest.importorskip("herdsight._core")

TOL = 1e-12
N = 10_000


@pytest.fixture(scope="module")
def angles():
    rng = np.random.default_rng(20261118)
    return np.column_stack(
        [
            rng.uniform(-np.pi, np.pi, N),
            rng.uniform(-1.5, 1.5, N),
            rng.uniform(-np.pi, np.pi, N),
        ]
    )


@pytest.mark.parametrize("name", ["rot_x", "rot_y", "rot_z"])
def test_axis_rotations_match(name, angles):
    py_f, cpp_f = getattr(fr, name), getattr(_core, name)
    worst = max(np.abs(cpp_f(a) - py_f(a)).max() for a in angles[:, 0])
    assert worst <= TOL


def test_R_wb_from_ypr_matches(angles):
    worst = max(np.abs(_core.R_wb_from_ypr(*a) - fr.R_wb_from_ypr(*a)).max() for a in angles)
    assert worst <= TOL


def test_ypr_from_R_wb_matches(angles):
    worst = 0.0
    for a in angles:
        R = fr.R_wb_from_ypr(*a)
        diff = np.abs(np.subtract(_core.ypr_from_R_wb(R), fr.ypr_from_R_wb(R))).max()
        worst = max(worst, diff)
    assert worst <= TOL


# Counting (Phase 3): pair features, logistic log odds and greedy joining.


def _random_candidates(rng, n_stills=6, max_det=8, n_pairs=400):
    counts = rng.integers(0, max_det, n_stills)
    rows = []
    for _ in range(n_pairs):
        i, j = sorted(rng.choice(n_stills, 2, replace=False))
        if counts[i] and counts[j]:
            rows.append((i, rng.integers(counts[i]), j, rng.integers(counts[j])))
    rows = sorted(set(rows))
    from herdsight.count.evidence import Candidates

    n = len(rows)
    arr = np.array(rows, int).reshape(-1, 4)
    c = Candidates(
        i=arr[:, 0],
        a=arr[:, 1],
        j=arr[:, 2],
        b=arr[:, 3],
        d_m=rng.uniform(0, 4, n),
        dt_s=rng.uniform(0, 200, n),
        geotag=rng.random(n) < 0.1,
        fixed_gate_m=rng.uniform(0.5, 3, n),
        size=rng.uniform(0, 0.5, n),
        low_score=rng.uniform(0, 0.7, n),
    )
    return [int(x) for x in counts], c


def _cpp_features(c, diffusion, kind):
    return _core.pair_features(
        c.i,
        c.a,
        c.j,
        c.b,
        c.d_m,
        c.dt_s,
        c.geotag.astype(float),
        c.fixed_gate_m,
        c.size,
        c.low_score,
        diffusion,
        kind,
    )


@pytest.mark.parametrize("kind", ["fixed", "movement"])
def test_pair_features_match(kind):
    from herdsight.count.evidence import MotionModel, features

    rng = np.random.default_rng(31)
    for _ in range(20):
        _, c = _random_candidates(rng)
        py = features(c, MotionModel(0.5, 0.004), kind)
        cpp = _cpp_features(c, 0.004, kind)
        assert np.abs(py - cpp).max() <= TOL


def test_log_odds_and_greedy_join_match():
    from herdsight.count.clustering import greedy_join
    from herdsight.count.evidence import Logistic

    rng = np.random.default_rng(32)
    for _ in range(20):
        counts, c = _random_candidates(rng)
        x = _cpp_features(c, 0.004, "fixed")
        m = Logistic(rng.normal(size=6), rng.uniform(0.5, 2, 6), rng.normal(size=7))
        lo_py, lo_cpp = m.log_odds(x), _core.log_odds(x, m.mean, m.std, m.coef)
        assert np.abs(lo_py - lo_cpp).max() <= 1e-9
        pairs = [
            ((int(c.i[k]), int(c.a[k])), (int(c.j[k]), int(c.b[k])), float(lo_py[k]))
            for k in range(len(lo_py))
        ]
        py_groups = greedy_join(counts, pairs)
        cpp_groups = _core.greedy_join(counts, c.i, c.a, c.j, c.b, lo_py)
        assert sorted(sorted(map(tuple, g)) for g in cpp_groups) == sorted(
            sorted(map(tuple, g)) for g in py_groups
        )


def test_counting_matches_on_recorded_flights():
    """The kept method on the 20 verified flights' recorded pair tables (storage drive)."""
    from herdsight.count.clustering import greedy_join
    from herdsight.utils.paths import storage_path

    try:
        path = storage_path("processed", "phase3") / "pairs_dfine.npz"
    except Exception:  # noqa: BLE001 - no storage drive configured
        pytest.skip("storage drive not configured")
    if not path.exists():
        pytest.skip("recorded pair tables not on this machine")
    import json

    d = np.load(path)
    counts = json.loads(str(d["counts"]))
    for key, n in counts.items():
        m = d["flight"] == key
        if not m.any():
            continue
        lo = d["log_odds_fixed"][m]
        i, a, j, b = (d[k][m].astype(int) for k in ("i", "a", "j", "b"))
        pairs = [
            ((int(i[k]), int(a[k])), (int(j[k]), int(b[k])), float(lo[k])) for k in range(len(lo))
        ]
        assert len(_core.greedy_join(n, i, a, j, b, lo)) == len(greedy_join(n, pairs)), key
