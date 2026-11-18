# Adapted from drone-ranger (https://github.com/LukeA77/drone-ranger),
# commit b4f4e4f, tests/test_ground_cov.py.
"""Propagated ground-point covariance (geometry/ground_cov.py) against finite differences and
known cases."""

import numpy as np
import pytest

from herdsight.geo.ground_cov import ground_covariance, ground_jacobian

F, CX, CY = 1259.0, 960.0, 540.0


def ground_offset_oblique(u, v, h, f, yaw, tilt_deg, cx=CX, cy=CY):
    """Independent reference: north/east offset of the flat-ground point seen at pixel (u, v)
    from a camera at height h, tilted tilt_deg below the horizon, heading yaw. Copied from
    drone-ranger's AU-AIR module (evaluate/auair_world.py, same commit) so the test keeps an
    implementation written separately from ground_cov."""
    x = (np.asarray(u, float) - cx) / f
    y = (np.asarray(v, float) - cy) / f
    t = np.radians(np.asarray(tilt_deg, float))
    c, s = np.cos(t), np.sin(t)
    fwd = c - s * y
    down = s + c * y
    right = x
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(down > 1e-9, np.asarray(h, float) / down, np.nan)
    fwd, right = k * fwd, k * right
    cy_, sy_ = np.cos(yaw), np.sin(yaw)
    return cy_ * fwd - sy_ * right, sy_ * fwd + cy_ * right


def _ground(p):
    u, v, tilt_rad, h, yaw = p
    return np.array(ground_offset_oblique(u, v, h, F, yaw, np.degrees(tilt_rad), CX, CY))


@pytest.mark.parametrize(
    "u, v, tilt, h, yaw",
    [
        (960, 540, 90, 30, 0.0),
        (300, 900, 45, 25, 0.7),
        (1700, 200, 20, 40, -2.0),
        (1000, 560, 8, 30, 1.2),
    ],
)
def test_jacobian_matches_finite_differences(u, v, tilt, h, yaw):
    off, jac = ground_jacobian(u, v, h, F, yaw, tilt, CX, CY)
    p = np.array([u, v, np.radians(tilt), h, yaw], float)
    np.testing.assert_allclose(off, _ground(p), rtol=1e-12)
    steps = np.array([1e-3, 1e-3, 1e-7, 1e-5, 1e-7])
    num = np.column_stack(
        [
            (_ground(p + d) - _ground(p - d)) / (2 * s)
            for s, d in zip(steps, np.diag(steps), strict=True)
        ]
    )
    np.testing.assert_allclose(jac, num, rtol=1e-5, atol=1e-8)


def test_nadir_pixel_error_scales_with_height_over_focal():
    _, cov = ground_covariance(960, 540, 30, F, 0.0, 90, CX, CY, 2.0, 2.0, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(np.sqrt(np.diag(cov)), [2 * 30 / F] * 2, rtol=1e-9)


def test_tilt_error_along_the_line_of_sight_grows_as_one_over_sine_squared():
    # image centre at depression d: ground distance h / tan d, d(dist)/d(tilt) = -h / sin^2 d
    for d in (10.0, 20.0, 45.0):
        _, cov = ground_covariance(960, 540, 30, F, 0.0, d, CX, CY, 0.0, 0.0, 1.0, 0.0, 0.0)
        expect = 30 / np.sin(np.radians(d)) ** 2 * np.radians(1.0)
        assert np.sqrt(cov[0, 0]) == pytest.approx(expect, rel=1e-9)  # north = along sight
        assert cov[1, 1] == pytest.approx(0.0, abs=1e-12)


def test_covariance_is_vectorised_and_nan_above_the_horizon():
    off, cov = ground_covariance(
        [960, 960], [540, 0], 30, F, 0.0, 10, CX, CY, 1.0, [1.0, 2.0], 0.5, 0.2, 1.0
    )
    assert cov.shape == (2, 2, 2) and np.isfinite(cov[0]).all()
    assert np.isnan(off[1]).all() and np.isnan(cov[1]).all()
    assert np.all(np.linalg.eigvalsh(cov[0]) >= 0)
