import math

import numpy as np
import pytest

from herdsight.count.ranges import fit, inside


def test_centre_is_median_log_ratio_and_corrects_bias():
    counts = np.array([110, 220, 55])
    truths = np.array([100, 200, 50])  # every count 10% high
    cal = fit(counts, truths, (0.68,))
    assert cal.centre == pytest.approx(math.log(1 / 1.1))
    assert cal.corrected(110) == pytest.approx(100)


def test_too_few_flights_give_an_unbounded_range():
    cal = fit([100] * 13, [90 + k for k in range(13)], (0.68, 0.90, 0.95))
    assert math.isfinite(cal.q[0.68]) and math.isfinite(cal.q[0.90])
    assert math.isinf(cal.q[0.95])
    assert cal.interval(100, 0.95) == (0.0, math.inf)


def test_coverage_on_exchangeable_flights():
    rng = np.random.default_rng(0)
    hits = []
    for _ in range(400):
        truth = rng.integers(20, 200, 15)
        count = np.maximum(1, np.round(truth * np.exp(rng.normal(0.05, 0.15, 15))))
        cal = fit(count[:14], truth[:14], (0.90,))
        lo, hi = cal.interval(count[14], 0.90)
        hits.append(inside(truth[14], lo, hi))
    assert np.mean(hits) >= 0.87


def test_inside_rounds_outwards():
    assert inside(141, 141.3, 144.6) and inside(145, 141.3, 144.6)
    assert not inside(146, 141.3, 144.6)
    assert inside(5, 0.0, math.inf)
