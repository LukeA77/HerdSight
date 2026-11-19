"""Full-resolution tiling of survey stills (D-8: no down-scaling for the official count).

A large still (e.g. 5280 x 3956) is cut into overlapping square tiles at native resolution. The
overlap must exceed the largest animal, so every animal lies wholly inside at least one tile.

Duplicates from the overlap are removed by *centre ownership*, not by non-maximum suppression:
each tile owns the central part of itself (the overlap split half-and-half with its neighbours,
and everything up to the image edge), and a detection is kept only by the tile that owns its
centre. Owned regions partition the image exactly, so each animal is counted once. Non-maximum
suppression across tiles is avoided because it deletes real animals standing close together
(the reason for oriented boxes, D-8).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


def tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    """Start offsets along one axis: step `tile - overlap`, last tile flush with the edge."""
    if tile <= 0 or not 0 <= overlap < tile:
        raise ValueError("need tile > 0 and 0 <= overlap < tile")
    if length <= tile:
        return [0]
    step = tile - overlap
    starts = list(range(0, length - tile, step))
    starts.append(length - tile)
    return starts


@dataclass(frozen=True)
class Tile:
    x0: int
    y0: int
    x1: int
    y1: int
    # owned region (half-open on the right/bottom, except at the image edge)
    ox0: float
    oy0: float
    ox1: float
    oy1: float

    def owns(self, cx: NDArray[np.float64], cy: NDArray[np.float64]) -> NDArray[np.bool_]:
        return (cx >= self.ox0) & (cx < self.ox1) & (cy >= self.oy0) & (cy < self.oy1)


def _owned_bounds(starts: list[int], tile: int, length: int) -> list[tuple[float, float]]:
    """Split each overlap at its midpoint; the first and last tiles own up to the edges."""
    bounds = []
    for i, s in enumerate(starts):
        lo = -np.inf if i == 0 else (starts[i - 1] + tile + s) / 2
        hi = np.inf if i == len(starts) - 1 else (s + tile + starts[i + 1]) / 2
        bounds.append((lo, hi))
    return bounds


def tile_grid(width: int, height: int, tile: int, overlap: int) -> list[Tile]:
    """All tiles covering the image, row by row, with their owned regions."""
    xs, ys = tile_starts(width, tile, overlap), tile_starts(height, tile, overlap)
    bx, by = _owned_bounds(xs, tile, width), _owned_bounds(ys, tile, height)
    return [
        Tile(x, y, min(x + tile, width), min(y + tile, height), ox[0], oy[0], ox[1], oy[1])
        for y, oy in zip(ys, by, strict=True)
        for x, ox in zip(xs, bx, strict=True)
    ]


def keep_owned(tile: Tile, boxes_xyxy: ArrayLike) -> NDArray[np.bool_]:
    """For boxes in full-image coordinates detected in `tile`: True where the tile owns the
    box centre."""
    b = np.asarray(boxes_xyxy, float).reshape(-1, 4)
    return tile.owns((b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2)


def boxes_in_tile(
    tile: Tile, boxes_xyxy: ArrayLike, min_visible: float = 0.5
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Training labels for one tile: boxes clipped to the tile, in tile coordinates.

    Returns (clipped boxes, mask of kept boxes). A box is kept if at least `min_visible` of its
    area lies inside the tile; animals cut off more than that are left out of this tile's labels
    (they are wholly inside a neighbouring tile, because the overlap exceeds the animal size).
    """
    b = np.asarray(boxes_xyxy, float).reshape(-1, 4)
    cx0 = np.clip(b[:, 0], tile.x0, tile.x1)
    cy0 = np.clip(b[:, 1], tile.y0, tile.y1)
    cx1 = np.clip(b[:, 2], tile.x0, tile.x1)
    cy1 = np.clip(b[:, 3], tile.y0, tile.y1)
    area = np.maximum(b[:, 2] - b[:, 0], 0) * np.maximum(b[:, 3] - b[:, 1], 0)
    vis = np.maximum(cx1 - cx0, 0) * np.maximum(cy1 - cy0, 0)
    keep = (area > 0) & (vis >= min_visible * area)
    out = np.column_stack([cx0 - tile.x0, cy0 - tile.y0, cx1 - tile.x0, cy1 - tile.y0])[keep]
    return out, keep
