import numpy as np
import pytest

from herdsight.count.evidence import (
    Candidates,
    MotionModel,
    Overlap,
    Still,
    auc,
    candidates,
    features,
    fit_logistic,
    match_to_labels,
)


def _still(t, centres, gsd=0.02, length=1.5, score=1.0):
    c = np.asarray(centres, float).reshape(-1, 2)
    return Still(
        time_s=t,
        gsd_m=gsd,
        centres_px=c,
        ground_m=c * gsd,
        length_m=np.full(len(c), length),
        score=np.full(len(c), score),
    )


def test_motion_model_recovers_floor_and_diffusion():
    rng = np.random.default_rng(0)
    dt = rng.uniform(0, 120, 4000)
    floor, diff = 0.4, 0.002
    d = np.sqrt(rng.chisquare(2, len(dt)) / 2 * (floor**2 + 4 * diff * dt))
    m = MotionModel.fit(d, dt)
    assert m.floor_m == pytest.approx(floor, rel=0.1)
    assert m.diffusion_m2_s == pytest.approx(diff, rel=0.2)
    assert m.allowance(np.array([0.0]))[0] == pytest.approx(m.floor_m)


def test_motion_model_never_negative():
    m = MotionModel.fit(np.full(10, 0.5), np.linspace(0, 100, 10))  # no growth with time
    assert m.diffusion_m2_s == pytest.approx(0.0, abs=1e-12)
    assert m.floor_m == pytest.approx(0.5)


def test_candidates_use_registration_then_geotag():
    s0 = _still(0.0, [[100, 100], [300, 100]])
    s1 = _still(4.0, [[110, 100], [600, 600]])
    shift = np.array([[1, 0, -10], [0, 1, 0], [0, 0, 1]], float)  # still 1 is 10 px right
    c = candidates([s0, s1], [Overlap(0, 1, shift)], gate_m=1.0)
    assert len(c) == 1 and (c.a[0], c.b[0]) == (0, 0)
    assert c.d_m[0] == pytest.approx(0.0)
    assert not c.geotag[0] and c.dt_s[0] == 4.0
    assert c.fixed_gate_m[0] == pytest.approx(0.75)  # half the median length
    g = candidates([s0, s1], [Overlap(0, 1, None)], geotag_gate_m=1.0)
    assert g.geotag.all() and g.d_m[0] == pytest.approx(0.2)  # 10 px * 0.02 m
    assert g.fixed_gate_m[0] == 3.0


def test_movement_allowance_forgives_distance_after_a_long_gap():
    s0 = _still(0.0, [[0, 0]])
    near = candidates([s0, _still(2.0, [[50, 0]])], [Overlap(0, 1, np.eye(3))])
    late = candidates([s0, _still(300.0, [[50, 0]])], [Overlap(0, 1, np.eye(3))])
    m = MotionModel(0.4, 0.002)
    z_near = features(near, m)[0, 0]
    z_late = features(late, m)[0, 0]
    assert z_late < z_near  # the same 1 m is less surprising five minutes later
    # with no grazing the two feature sets agree
    still = MotionModel(0.4, 0.0)
    assert features(late, still)[0, 0] == features(late, still, "fixed")[0, 0]
    # the fixed gate does not care about time
    assert features(near, m, "fixed")[0, 0] == features(late, m, "fixed")[0, 0]


def test_competition_marks_mutual_closest_and_margin():
    s0 = _still(0.0, [[0, 0], [100, 0]])
    s1 = _still(2.0, [[5, 0], [60, 0]])
    c = candidates([s0, s1], [Overlap(0, 1, np.eye(3))], gate_m=4.0)
    f = features(c, MotionModel(0.4, 0.0))
    rows = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(c.a, c.b, strict=True))}
    assert f[rows[(0, 0)], 2] == 1.0  # 0-0 are each other's closest
    assert f[rows[(0, 1)], 2] == 0.0
    assert f[rows[(0, 0)], 1] > 0  # its nearest rival is further away
    assert f[rows[(0, 1)], 1] < 0  # this pair has a closer rival


def test_logistic_separates_and_auc():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2000, 2))
    y = (x[:, 0] - 0.5 * x[:, 1] + rng.normal(scale=0.5, size=2000)) > 0
    m = fit_logistic(x, y)
    assert m.coef[1] > 0 and m.coef[2] < 0
    assert auc(m.log_odds(x), y) > 0.9
    assert auc(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0, 0, 1, 1])) == 1.0
    assert auc(np.array([1.0, 1.0]), np.array([0, 1])) == 0.5


def test_match_to_labels_one_to_one_within_half_length():
    det = np.array([[0, 0], [1, 0], [50, 50]], float)
    lab = np.array([[0.5, 0], [100, 100]], float)
    m = match_to_labels(det, lab, np.array([10.0, 10.0]))
    assert sorted(m.tolist()) == [-1, -1, 0]


def test_empty_inputs():
    c = candidates([_still(0.0, []), _still(1.0, [[0, 0]])], [Overlap(0, 1, np.eye(3))])
    assert isinstance(c, Candidates) and len(c) == 0
