import math

import numpy as np
import pytest

from herdsight.geo.nadir import (
    camera_ned,
    focal_px_from_35mm,
    ground_sample_distance_m,
    pixel_to_ground,
)


def test_focal_from_35mm_and_gsd():
    # 24 mm equivalent on a 5280 x 3956 image: f = 24 * 6597.6 / 43.27
    f = focal_px_from_35mm(24, 5280, 3956)
    assert f == pytest.approx(24 * math.hypot(5280, 3956) / 43.27)
    assert ground_sample_distance_m(100, f) == pytest.approx(100 / f)


def test_image_centre_is_below_the_camera():
    np.testing.assert_allclose(
        pixel_to_ground([2640], [1978], 100, 3000, 37, 5280, 3956), [[0, 0]], atol=1e-9
    )


@pytest.mark.parametrize("yaw", [0.0, 90.0, 180.0, -45.0])
def test_top_of_image_faces_the_yaw(yaw):
    f, h = 3000.0, 100.0
    off = pixel_to_ground([2640], [1978 - 300], h, f, yaw, 5280, 3956)[0]
    d = 300 * h / f
    np.testing.assert_allclose(
        off, [d * math.cos(math.radians(yaw)), d * math.sin(math.radians(yaw))], atol=1e-9
    )


def test_right_of_image_is_right_of_heading_and_mirror_flips_it():
    f, h = 3000.0, 100.0
    east = pixel_to_ground([2640 + 300], [1978], h, f, 0.0, 5280, 3956)[0]
    np.testing.assert_allclose(east, [0, 10], atol=1e-9)
    west = pixel_to_ground([2640 + 300], [1978], h, f, 0.0, 5280, 3956, mirror=True)[0]
    np.testing.assert_allclose(west, [0, -10], atol=1e-9)


def test_camera_ned_small_offsets():
    ne = camera_ned([47.0 + 1e-4], [-1.5], 47.0, -1.5)[0]
    assert ne[0] == pytest.approx(11.12, abs=0.05) and abs(ne[1]) < 1e-6


def test_ground_to_pixel_inverts_pixel_to_ground():
    from herdsight.geo.nadir import ground_to_pixel

    u, v = np.array([100.0, 2640.0, 5000.0]), np.array([300.0, 1978.0, 3900.0])
    for yaw in (0.0, 33.0, -120.0):
        ne = pixel_to_ground(u, v, 87.0, 3600.0, yaw, 5280, 3956)
        back = ground_to_pixel(ne[:, 0], ne[:, 1], 87.0, 3600.0, yaw, 5280, 3956)
        np.testing.assert_allclose(back, np.column_stack([u, v]), atol=1e-6)
