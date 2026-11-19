"""One-to-one pairing of animal positions between two overlapping images."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linear_sum_assignment


def match_points(pa: ArrayLike, pb: ArrayLike, gate: float) -> list[tuple[int, int, float]]:
    """Pairs (index in a, index in b, distance) from a minimum-total-distance one-to-one
    assignment, keeping only pairs no further apart than `gate`. Both inputs are (N, 2) points in
    one frame (pixels of image a, or ground metres)."""
    a = np.asarray(pa, float).reshape(-1, 2)
    b = np.asarray(pb, float).reshape(-1, 2)
    if len(a) == 0 or len(b) == 0:
        return []
    dist = np.linalg.norm(a[:, None] - b[None], axis=2)
    r, c = linear_sum_assignment(np.where(dist <= gate, dist, 1e9))
    return [
        (int(i), int(j), float(dist[i, j])) for i, j in zip(r, c, strict=True) if dist[i, j] <= gate
    ]
