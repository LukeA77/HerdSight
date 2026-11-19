"""Upright-box AP (pycocotools) and count metrics on cases with known answers."""

import math

import numpy as np
import pytest

from herdsight.evaluate.counts import (
    count_errors,
    count_interval,
    coverage,
    coverage_by_band,
    fit_count_conformal,
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
