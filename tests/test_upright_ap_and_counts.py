"""Upright-box AP (pycocotools) and count metrics on cases with known answers."""

import math

import numpy as np
import pytest

from herdsight.evaluate.counts import (
    band_interval,
    count_errors,
    count_interval,
    coverage,
    coverage_by_band,
    fit_band_conformal,
    fit_count_conformal,
    predicted_band,
)
from herdsight.evaluate.upright_ap import ImageBoxes, obb_to_upright, upright_ap


def test_obb_to_upright_axis_aligned_and_rotated():
    np.testing.assert_allclose(obb_to_upright([[10, 20, 8, 4, 0.0]]), [[6, 18, 14, 22]])
    # 90 degrees swaps width and height
    np.testing.assert_allclose(
        obb_to_upright([[10, 20, 8, 4, math.pi / 2]]), [[8, 16, 12, 24]], atol=1e-12
    )
    # 45 degrees: half-extent (w + h) / (2 sqrt 2) in both axes
    r = (8 + 4) / (2 * math.sqrt(2))
    np.testing.assert_allclose(
        obb_to_upright([[0, 0, 8, 4, math.pi / 4]]), [[-r, -r, r, r]], atol=1e-12
    )


def test_perfect_boxes_give_ap_one():
    boxes = [[0, 0, 40, 20], [100, 100, 130, 160]]
    r = upright_ap([ImageBoxes.make(boxes, [0.9, 0.8], boxes)])
    assert r["ap50"] == pytest.approx(1.0) and r["ap50_95"] == pytest.approx(1.0)


def test_missed_and_spurious_boxes_lower_ap():
    true = [[0, 0, 40, 20], [100, 100, 130, 160]]
    r = upright_ap([ImageBoxes.make([[0, 0, 40, 20], [500, 500, 520, 520]], [0.9, 0.95], true)])
    assert 0.0 < r["ap50"] < 1.0


def test_more_than_100_animals_are_all_scored():
    # COCO's default would keep only 100 detections per image; a big herd must not be truncated.
    true = np.array([[i * 30, 0, i * 30 + 20, 20] for i in range(250)], float)
    r = upright_ap([ImageBoxes.make(true, np.linspace(1, 0.5, 250), true)])
    assert r["ap50"] == pytest.approx(1.0)
    assert r["max_dets"] >= 250


def test_no_truth_is_nan_and_no_predictions_is_zero():
    assert math.isnan(upright_ap([ImageBoxes.make(np.zeros((0, 4)), [], np.zeros((0, 4)))])["ap50"])
    assert upright_ap([ImageBoxes.make(np.zeros((0, 4)), [], [[0, 0, 10, 10]])])["ap50"] == 0.0


def test_count_errors_known_values():
    r = count_errors([10, 12, 0], [10, 10, 1])
    assert r["mae"] == pytest.approx(1.0)
    assert r["rmse"] == pytest.approx(math.sqrt(5 / 3))
    assert r["mean_signed_error"] == pytest.approx(1 / 3)
    assert r["exact_share"] == pytest.approx(1 / 3)


def test_count_interval_rounds_outwards_and_clips_at_zero():
    lo, hi = count_interval([0.0, 10.0], q=1.0)
    np.testing.assert_allclose(lo, [0, math.floor(10 - math.sqrt(11))])
    np.testing.assert_allclose(hi, [1, math.ceil(10 + math.sqrt(11))])
    lo, hi = count_interval([5.0], q=math.inf)
    assert lo[0] == 0 and math.isinf(hi[0])


def test_conformal_count_intervals_reach_nominal_coverage_on_exchangeable_data():
    rng = np.random.default_rng(0)
    true = rng.poisson(15, 4000)
    pred = np.maximum(true + rng.normal(0, 2, true.size), 0)
    q = fit_count_conformal(pred[:2000], true[:2000])
    cov = coverage(pred[2000:], true[2000:], q)
    for lv in (0.68, 0.90, 0.95):
        assert cov[f"coverage_{lv:.2f}"] >= lv - 0.03


def test_coverage_by_band_reports_each_band():
    rows = coverage_by_band([0, 3, 10, 30], [0, 3, 12, 25], {0.9: 1.0})
    assert [r["n"] for r in rows] == [1, 1, 1, 1]
    assert "coverage_0.90" in rows[0]


def test_iou_matrix_and_matching():
    from herdsight.evaluate.upright_ap import iou_matrix, match_iou

    m = iou_matrix([[0, 0, 10, 10], [5, 0, 15, 10]], [[0, 0, 10, 10]])
    np.testing.assert_allclose(m[:, 0], [1.0, 50 / 150])
    assert match_iou([[0, 0, 10, 10], [5, 0, 15, 10]], [[0, 0, 10, 10]]) == 1
    assert match_iou(np.zeros((0, 4)), [[0, 0, 1, 1]]) == 0


def test_predicted_band_uses_lower_edges():
    b = predicted_band([0, 0.4, 1, 4.9, 5, 19, 20, 300, -1], (0, 1, 5, 20))
    assert b.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 0]
    with pytest.raises(ValueError):
        predicted_band([1], (1, 5))  # must cover zero
    with pytest.raises(ValueError):
        predicted_band([1], (0, 5, 5))


def test_band_conformal_covers_each_band_where_pooled_does_not():
    # synthetic: large herds are miscounted far more (in sqrt(c + 1) units) than small ones
    rng = np.random.default_rng(3)

    def make(n):
        c = np.concatenate([rng.integers(0, 5, n), rng.integers(20, 60, n // 4)]).astype(float)
        spread = np.where(c >= 20, 3.0, 0.4) * np.sqrt(c + 1)
        return c, np.maximum(np.round(c + spread * rng.normal(size=c.size)), 0)

    edges = (0, 1, 5, 20)
    c_cal, t_cal = make(4000)
    c_new, t_new = make(4000)
    model = fit_band_conformal(c_cal, t_cal, edges, (0.9,))
    big = c_new >= 20
    lo, hi = band_interval(c_new, model, 0.9)
    inside = (t_new >= lo) & (t_new <= hi)
    assert inside[big].mean() >= 0.87 and inside[~big].mean() >= 0.87
    q = fit_count_conformal(c_cal, t_cal, (0.9,))[0.9]
    plo, phi = count_interval(c_new, q)
    assert ((t_new >= plo) & (t_new <= phi))[big].mean() < 0.8  # pooled under-covers big herds


def test_band_conformal_reports_unbounded_when_a_band_is_too_small():
    model = fit_band_conformal([0.0, 0.0, 3.0], [0.0, 1.0, 3.0], (0, 1, 5, 20), (0.9,))
    assert [b["n"] for b in model["bands"]] == [2, 1, 0, 0]
    assert all(math.isinf(b["q"][0.9]) for b in model["bands"])
    lo, hi = band_interval([25.0], model, 0.9)
    assert lo[0] == 0 and math.isinf(hi[0])
