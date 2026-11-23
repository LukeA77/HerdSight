"""Oriented tile labels and the eye-check result."""

import numpy as np
import pytest

from herdsight.labels.obb_tiles import attach_obbs, enclosing_upright, obb_quad, parse_tile_name
from herdsight.labels.sam_final import review_outcome


def test_tile_name_is_parsed():
    t = parse_tile_name("Derval__DJI_2023_012__DJI_0007_V__448_96.jpg")
    assert t == ("Derval", "DJI_2023_012", "DJI_0007_V", 448, 96)


def test_bad_tile_name_is_refused():
    with pytest.raises(ValueError):
        parse_tile_name("Derval_0007.jpg")


FULL = [[100.0, 100.0, 180.0, 140.0], [700.0, 90.0, 760.0, 170.0]]
OBBS = [[140.0, 120.0, 85.0, 30.0, 0.2], [730.0, 130.0, 80.0, 28.0, -1.0]]


def test_obb_moves_into_tile_coordinates():
    out = attach_obbs([[36.0, 36.0, 80.0, 40.0]], 64, 64, 640, 640, FULL, OBBS)
    assert out[0] == pytest.approx([76.0, 56.0, 85.0, 30.0, 0.2])


def test_clipped_label_finds_its_animal():
    # the second animal, cut by the tile's right edge at x = 740
    out = attach_obbs([[600.0, 90.0, 40.0, 80.0]], 100, 0, 640, 640, FULL, OBBS)
    assert out[0] == pytest.approx([630.0, 130.0, 80.0, 28.0, -1.0])


def test_identical_boxes_each_use_a_different_animal():
    full = [FULL[0], FULL[0]]
    obbs = [OBBS[0], [140.0, 120.0, 70.0, 20.0, 0.9]]
    out = attach_obbs([[100.0, 100.0, 80.0, 40.0]] * 2, 0, 0, 640, 640, full, obbs)
    assert sorted(out[:, 2].tolist()) == [70.0, 85.0]


def test_label_without_an_animal_is_an_error():
    with pytest.raises(ValueError):
        attach_obbs([[10.0, 10.0, 20.0, 20.0]], 0, 0, 640, 640, FULL, OBBS)


@pytest.mark.parametrize("angle", [-1.4, -0.6, 0.0, 0.5, 1.2])
def test_quad_has_the_box_area_and_enclosing_box_contains_it(angle):
    obb = [50.0, 60.0, 40.0, 15.0, angle]
    q = np.array(obb_quad(obb)).reshape(4, 2)
    x, y = q[:, 0], q[:, 1]
    area = abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2
    assert area == pytest.approx(600.0)
    x0, y0, x1, y1 = enclosing_upright(obb)[0]
    assert q[:, 0].min() == pytest.approx(x0) and q[:, 0].max() == pytest.approx(x1)
    assert q[:, 1].min() == pytest.approx(y0) and q[:, 1].max() == pytest.approx(y1)


def _key(n=10):
    return {f"{i:03d}": {"fold": "fold0", "flagged": i < 2} for i in range(n)}


def test_review_passes_at_the_minimum():
    key = _key()
    answers = {k: "good" for k in key}
    answers["000"] = "bad"
    out = review_outcome(answers, key, 0.9)
    assert out["passed"] and out["accepted_share"] == pytest.approx(0.9)
    assert out["by_stratum"]["fold0_flagged"] == {"good": 1, "bad": 1, "unsure": 0}


def test_not_sure_does_not_count_as_accepted():
    key = _key()
    answers = {k: "good" for k in key}
    answers["000"] = answers["001"] = "unsure"
    assert not review_outcome(answers, key, 0.9)["passed"]


def test_unfinished_review_is_refused():
    key = _key()
    answers = {k: "good" for k in list(key)[:-1]}
    with pytest.raises(ValueError):
        review_outcome(answers, key, 0.9)
