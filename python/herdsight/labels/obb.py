"""Oriented boxes from segmentation masks (brief 8.4: upright labels -> oriented training labels).

The segmentation model (Phase 2) outlines the animal inside its upright box; the tightest rotated
rectangle around the outline becomes the oriented box. Oriented boxes are (cx, cy, w, h, angle)
with w >= h (w along the body), angle in radians in [-pi/2, pi/2), measured from the image x axis
towards the image y axis (clockwise on screen, since image y points down).

Generated labels are for training only; they are never used to score a model (brief 8.4, 9),
except the hand-verified subset of section 9.3.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


def normalise_obb(cx: float, cy: float, w: float, h: float, angle: float) -> NDArray[np.float64]:
    """Canonical form: w >= h, angle in [-pi/2, pi/2)."""
    if h > w:
        w, h, angle = h, w, angle + math.pi / 2
    angle = (angle + math.pi / 2) % math.pi - math.pi / 2
    return np.array([cx, cy, w, h, angle], float)


def mask_to_obb(mask: ArrayLike, min_pixels: int = 10) -> NDArray[np.float64] | None:
    """Tightest rotated rectangle around the largest connected region of a binary mask.

    Returns None if the mask has fewer than `min_pixels` pixels in its largest region (the
    segmentation failed; such labels are flagged for review, never guessed)."""
    import cv2

    m = (np.asarray(mask) > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[big, cv2.CC_STAT_AREA] < min_pixels:
        return None
    ys, xs = np.nonzero(lab == big)
    # Pixel corners, not centres, so a one-pixel-wide region has width 1.
    pts = np.concatenate(
        [np.column_stack([xs + dx, ys + dy]) for dx in (0, 1) for dy in (0, 1)]
    ).astype(np.float32)
    (cx, cy), (w, h), deg = cv2.minAreaRect(pts)
    return normalise_obb(cx, cy, w, h, math.radians(deg))


def obb_corners(obb: ArrayLike) -> NDArray[np.float64]:
    """The four corners (4, 2) of an oriented box, in order around the box."""
    cx, cy, w, h, a = np.asarray(obb, float)
    u = np.array([math.cos(a), math.sin(a)])
    v = np.array([-math.sin(a), math.cos(a)])
    c = np.array([cx, cy])
    return np.array(
        [
            c - u * w / 2 - v * h / 2,
            c + u * w / 2 - v * h / 2,
            c + u * w / 2 + v * h / 2,
            c - u * w / 2 + v * h / 2,
        ]
    )


def rotated_iou(a: ArrayLike, b: ArrayLike) -> float:
    """Intersection over union of two oriented boxes (for the hand-verified subset, brief 9.3)."""
    import cv2

    pa = obb_corners(a).astype(np.float32)
    pb = obb_corners(b).astype(np.float32)
    inter_type, inter = cv2.intersectConvexConvex(pa, pb)
    inter_area = float(inter_type) if inter is not None else 0.0
    area_a = float(np.asarray(a)[2] * np.asarray(a)[3])
    area_b = float(np.asarray(b)[2] * np.asarray(b)[3])
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0
