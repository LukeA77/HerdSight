"""Ground positions of pixels in nadir survey stills (D-9 step 1, for geotagged drone stills).

For a camera looking straight down from height h above flat ground, with focal length f pixels
and principal point (cx, cy) at the image centre, a pixel (u, v) lies on the ground at

    forward = -(v - cy) * h / f,   right = (u - cx) * h / f

relative to the point below the camera, where "forward" is the direction the top of the image
faces. The top of the image faces the gimbal yaw (degrees clockwise from north) for DJI nadir
stills; this is checked against the data by `scripts/check_geolocation.py`, not assumed. The
geometry is the nadir case of `herdsight.geo.ground_cov.ground_jacobian` (tilt 90 degrees).

The focal length in pixels comes from the 35 mm-equivalent focal length in EXIF:
f_px = f35 * image_diagonal_px / 43.27 mm (the full-frame diagonal).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from herdsight.geo.frames import geodetic_to_ned
from herdsight.geo.ground_cov import ground_jacobian

FULL_FRAME_DIAGONAL_MM = 43.27


def focal_px_from_35mm(f35_mm: float, width: int, height: int) -> float:
    """Focal length in pixels from the 35 mm-equivalent focal length (diagonal convention)."""
    return f35_mm * math.hypot(width, height) / FULL_FRAME_DIAGONAL_MM


def ground_sample_distance_m(height_m: float, f_px: float) -> float:
    """Metres per pixel at the image centre of a nadir camera."""
    return height_m / f_px


def pixel_to_ground(
    u: ArrayLike,
    v: ArrayLike,
    height_m: float,
    f_px: float,
    yaw_deg: float,
    width: int,
    image_height: int,
    mirror: bool = False,
) -> NDArray[np.float64]:
    """(north, east) offsets in metres of pixels from the point below the camera, shape (n, 2).

    `mirror` flips the image left-right; it exists only so the convention can be tested."""
    u = np.asarray(u, float).reshape(-1)
    v = np.asarray(v, float).reshape(-1)
    cx, cy = width / 2.0, image_height / 2.0
    if mirror:
        u = 2 * cx - u
    off, _ = ground_jacobian(u, v, height_m, f_px, math.radians(yaw_deg), 90.0, cx, cy)
    return off.reshape(-1, 2)


def camera_ned(lat: ArrayLike, lon: ArrayLike, lat0: float, lon0: float) -> NDArray[np.float64]:
    """(north, east) of camera positions relative to an origin, ignoring height (shape (n, 2))."""
    ned = geodetic_to_ned(lat, lon, np.zeros_like(np.asarray(lat, float)), lat0, lon0, 0.0)
    return np.asarray(ned)[..., :2].reshape(-1, 2)


def ground_to_pixel(
    north: ArrayLike,
    east: ArrayLike,
    height_m: float,
    f_px: float,
    yaw_deg: float,
    width: int,
    image_height: int,
) -> NDArray[np.float64]:
    """Inverse of `pixel_to_ground` (unmirrored): pixels (n, 2) of ground offsets from the point
    below the camera."""
    n = np.asarray(north, float).reshape(-1)
    e = np.asarray(east, float).reshape(-1)
    psi = math.radians(yaw_deg)
    fwd = n * math.cos(psi) + e * math.sin(psi)
    right = -n * math.sin(psi) + e * math.cos(psi)
    return np.column_stack(
        [width / 2.0 + right * f_px / height_m, image_height / 2.0 - fwd * f_px / height_m]
    )
