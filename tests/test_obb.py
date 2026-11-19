import math

import numpy as np
import pytest

from herdsight.evaluate.upright_ap import obb_to_upright
from herdsight.labels.obb import mask_to_obb, normalise_obb, obb_corners, rotated_iou


def _rect_mask(cx, cy, w, h, angle, size=200):
    yy, xx = np.mgrid[0:size, 0:size] + 0.5
    dx, dy = xx - cx, yy - cy
    u = dx * math.cos(angle) + dy * math.sin(angle)
    v = -dx * math.sin(angle) + dy * math.cos(angle)
    return (np.abs(u) <= w / 2) & (np.abs(v) <= h / 2)


@pytest.mark.parametrize("angle_deg", [0, 15, 33, 60, 89, -40])
def test_mask_to_obb_recovers_a_rotated_animal_shape(angle_deg):
    a = math.radians(angle_deg)
    obb = mask_to_obb(_rect_mask(100, 90, 80, 30, a))
    assert obb is not None
    cx, cy, w, h, got = obb
    assert cx == pytest.approx(100, abs=1.0) and cy == pytest.approx(90, abs=1.0)
    assert w == pytest.approx(80, rel=0.05) and h == pytest.approx(30, rel=0.08)
    diff = (got - normalise_obb(0, 0, 80, 30, a)[4] + math.pi / 2) % math.pi - math.pi / 2
    assert abs(diff) < math.radians(2)


def test_largest_region_is_used_and_tiny_masks_are_refused():
    m = _rect_mask(60, 60, 60, 20, 0.0) | _rect_mask(170, 170, 6, 6, 0.0)
    obb = mask_to_obb(m)
    assert obb[0] == pytest.approx(60, abs=1)
    assert mask_to_obb(np.zeros((50, 50))) is None
    tiny = np.zeros((50, 50))
    tiny[10, 10] = 1
    assert mask_to_obb(tiny) is None


def test_normalise_puts_long_side_first_and_wraps_angle():
    o = normalise_obb(0, 0, 10, 40, 0.0)
    assert o[2] == 40 and o[3] == 10 and o[4] == pytest.approx(-math.pi / 2)
    assert -math.pi / 2 <= normalise_obb(0, 0, 40, 10, 3.5)[4] < math.pi / 2


def test_corners_and_upright_hull_agree():
    obb = [50, 40, 30, 10, math.radians(30)]
    c = obb_corners(obb)
    hull = [c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max()]
    np.testing.assert_allclose(obb_to_upright([obb])[0], hull, atol=1e-9)


def test_rotated_iou_known_values():
    a = [0, 0, 20, 10, 0.0]
    assert rotated_iou(a, a) == pytest.approx(1.0, abs=1e-6)
    assert rotated_iou(a, [10, 0, 20, 10, 0.0]) == pytest.approx(1 / 3, abs=1e-6)
    assert rotated_iou(a, [100, 0, 20, 10, 0.0]) == 0.0
    # Same box rotated by 90 degrees about its centre: intersection 10 x 10 = 100, union 300
    assert rotated_iou(a, [0, 0, 20, 10, math.pi / 2]) == pytest.approx(100 / 300, abs=1e-6)
