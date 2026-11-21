"""Centre-matching scores on hand-made cases with known answers."""

import math

import numpy as np
import pytest

from herdsight.evaluate.centre_match import (
    COCO_THRESHOLDS,
    ImageDetections,
    ap_mean_subset,
    average_precision,
    box_centres,
    match_hungarian,
    match_table,
    short_side,
    similarity_matrix,
)

KAPPA = 0.25


def test_similarity_is_one_at_zero_distance_and_follows_the_formula():
    s = similarity_matrix([[0, 0], [3, 4]], [[0, 0]], [20.0], KAPPA)
    assert s[0, 0] == pytest.approx(1.0)
    assert s[1, 0] == pytest.approx(math.exp(-25 / (2 * (KAPPA * 20) ** 2)))


def test_similarity_scales_with_animal_size():
    # The same 5 px offset matters less for a bigger animal.
    small = similarity_matrix([[5, 0]], [[0, 0]], [10.0], KAPPA)[0, 0]
    big = similarity_matrix([[5, 0]], [[0, 0]], [40.0], KAPPA)[0, 0]
    assert big > small


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        similarity_matrix([[0, 0]], [[0, 0]], [10.0], 0.0)
    with pytest.raises(ValueError):
        similarity_matrix([[0, 0]], [[0, 0]], [0.0], KAPPA)
    with pytest.raises(ValueError):
        similarity_matrix([[0, 0]], [[0, 0], [1, 1]], [10.0], KAPPA)


def test_box_helpers():
    boxes = [[10, 20, 50, 30]]
    np.testing.assert_allclose(box_centres(boxes), [[30, 25]])
    np.testing.assert_allclose(short_side(boxes), [10])


def test_centre_of_rotated_box_equals_centre_of_its_upright_hull():
    # The reason centre matching is orientation-independent (brief 9.1).
    c, w, h, a = np.array([100.0, 60.0]), 40.0, 12.0, np.radians(33)
    corners = np.array([[-w, -h], [w, -h], [w, h], [-w, h]]) / 2
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = corners @ rot.T + c
    hull = [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()]
    np.testing.assert_allclose(box_centres([hull])[0], c)


def test_hungarian_beats_greedy_on_a_crossing_case():
    # Prediction 0 is slightly closer to truth 1 than to truth 0, so greedy nearest matching would
    # take truth 1 and leave prediction 1 unmatched; the optimal matching pairs both.
    true = [[0, 0], [10, 0]]
    pred = [[5.5, 0], [16, 0]]
    m = match_hungarian(pred, true, [40.0, 40.0], KAPPA, threshold=0.1)
    assert m.tp == 2 and m.fp == 0 and m.fn == 0
    assert sorted(map(tuple, m.pairs.tolist())) == [(0, 0), (1, 1)]


def test_hungarian_drops_pairs_below_threshold():
    m = match_hungarian([[100, 100]], [[0, 0]], [20.0], KAPPA, threshold=0.5)
    assert (m.tp, m.fp, m.fn) == (0, 1, 1)


def test_hungarian_empty_cases():
    m = match_hungarian(np.zeros((0, 2)), [[0, 0]], [10.0], KAPPA)
    assert (m.tp, m.fp, m.fn) == (0, 0, 1)
    m = match_hungarian([[0, 0]], np.zeros((0, 2)), np.zeros(0), KAPPA)
    assert (m.tp, m.fp, m.fn) == (0, 1, 0)


def _img(pred, scores, true, sigma=20.0):
    true = np.asarray(true, float).reshape(-1, 2)
    return ImageDetections.make(pred, scores, true, np.full(len(true), sigma))


def test_perfect_predictions_give_ap_one():
    r = average_precision([_img([[0, 0], [50, 50]], [0.9, 0.8], [[0, 0], [50, 50]])], KAPPA)
    assert r["ap_mean"] == pytest.approx(1.0)
    assert len(r["ap"]) == len(COCO_THRESHOLDS) == 10


def test_far_predictions_give_ap_zero():
    r = average_precision([_img([[500, 500]], [0.9], [[0, 0]])], KAPPA)
    assert r["ap_mean"] == pytest.approx(0.0)


def test_duplicate_prediction_is_a_false_positive():
    # Two predictions on one animal: recall reaches 1 at precision 1 (first, higher score), so the
    # duplicate ranked below it does not reduce interpolated AP...
    r = average_precision([_img([[0, 0], [1, 0]], [0.9, 0.8], [[0, 0]])], KAPPA)
    assert r["ap_mean"] == pytest.approx(1.0)
    # ...but ranked above it, it does: precision at full recall is 1/2.
    r = average_precision([_img([[60, 0], [0, 0]], [0.9, 0.8], [[0, 0]])], KAPPA)
    assert r["ap"][0] == pytest.approx(0.5)


def test_false_positive_on_an_empty_image_counts():
    hit = _img([[0, 0]], [0.5], [[0, 0]])
    empty_fp = _img([[10, 10]], [0.9], np.zeros((0, 2)))
    r = average_precision([hit, empty_fp], KAPPA)
    assert r["ap"][0] == pytest.approx(0.5)  # the empty-image FP outranks the true hit
    assert r["n_true"] == 1 and r["n_pred"] == 2


def test_partial_recall_known_value():
    # Two animals, one found: precision 1 up to recall 0.5, then nothing -> 51 of 101 points.
    r = average_precision([_img([[0, 0]], [0.9], [[0, 0], [100, 100]])], KAPPA)
    assert r["ap"][0] == pytest.approx(51 / 101)


def test_stricter_thresholds_reject_offset_predictions():
    # Offset so that similarity is about 0.7: counted at 0.50-0.70, missed above.
    sigma, d = 20.0, None
    d = KAPPA * sigma * math.sqrt(-2 * math.log(0.7))
    r = average_precision([_img([[d, 0]], [0.9], [[0, 0]], sigma)], KAPPA)
    ap = dict(zip(r["thresholds"], r["ap"], strict=True))
    assert ap[0.5] == pytest.approx(1.0) and ap[0.65] == pytest.approx(1.0)
    assert ap[0.75] == pytest.approx(0.0) and ap[0.95] == pytest.approx(0.0)


def test_no_truth_anywhere_is_nan():
    r = average_precision([_img([[0, 0]], [0.9], np.zeros((0, 2)))], KAPPA)
    assert math.isnan(r["ap_mean"])


def test_ap_from_match_table_equals_full_ap_and_handles_resampling():
    hit = _img([[0, 0]], [0.5], [[0, 0]])
    empty_fp = _img([[10, 10]], [0.9], np.zeros((0, 2)))
    miss = _img(np.zeros((0, 2)), [], [[100, 100]])
    imgs = [hit, empty_fp, miss]
    table = match_table(imgs, KAPPA)
    assert [t.shape for t in table] == [(1, 10), (1, 10), (0, 10)]
    full = average_precision(imgs, KAPPA)["ap_mean"]
    assert ap_mean_subset(imgs, table, [0, 1, 2]) == pytest.approx(full)
    # a resample that repeats one image and drops another equals AP on that multiset of images
    assert ap_mean_subset(imgs, table, [0, 0, 2]) == pytest.approx(
        average_precision([hit, hit, miss], KAPPA)["ap_mean"]
    )
    assert math.isnan(ap_mean_subset(imgs, table, [1]))  # no truth in the sample
