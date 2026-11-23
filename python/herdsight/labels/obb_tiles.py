"""Oriented training labels for the 640 px tiles (Phase 2, step G7).

A tile's upright labels are the full-image labels clipped to the tile. Each one is matched back to
its full-image animal, whose oriented box is then moved into tile coordinates. The oriented box is
not clipped: part of an animal cut by the tile edge may lie outside the tile.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


class TileName(NamedTuple):
    farm: str
    flight: str
    stem: str
    x0: int
    y0: int


def parse_tile_name(name: str) -> TileName:
    """Farm__flight__imagestem__x0_y0.jpg, as written by scripts/make_training_tiles.py."""
    base = name.rsplit(".", 1)[0]
    parts = base.split("__")
    if len(parts) != 4:
        raise ValueError(f"not a tile name: {name}")
    x0, y0 = parts[3].split("_")
    return TileName(parts[0], parts[1], parts[2], int(x0), int(y0))


def attach_obbs(
    tile_boxes_xywh: ArrayLike,
    x0: int,
    y0: int,
    tile_w: int,
    tile_h: int,
    full_xyxy: ArrayLike,
    full_obbs: ArrayLike,
    tol: float = 0.03,
) -> NDArray[np.float64]:
    """The oriented box (cx, cy, w, h, angle) in tile pixels for each upright tile label.

    Each full-image animal is used at most once. A label with no match within `tol` pixels is an
    error, because a wrong pairing would silently teach the model a wrong box.
    """
    boxes = np.asarray(tile_boxes_xywh, float).reshape(-1, 4)
    full = np.asarray(full_xyxy, float).reshape(-1, 4)
    obbs = np.asarray(full_obbs, float).reshape(-1, 5)
    if len(full) != len(obbs):
        raise ValueError("one oriented box per full-image box")
    clipped = np.column_stack(
        [
            np.clip(full[:, 0], x0, x0 + tile_w) - x0,
            np.clip(full[:, 1], y0, y0 + tile_h) - y0,
            np.clip(full[:, 2], x0, x0 + tile_w) - x0,
            np.clip(full[:, 3], y0, y0 + tile_h) - y0,
        ]
    )
    wanted = np.column_stack(
        [boxes[:, 0], boxes[:, 1], boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 3]]
    )
    used = np.zeros(len(full), bool)
    out = np.zeros((len(boxes), 5))
    for i, box in enumerate(wanted):
        error = np.abs(clipped - box).max(axis=1) if len(full) else np.zeros(0)
        error = np.where(used, np.inf, error)
        if not len(error) or error.min() > tol:
            raise ValueError(f"tile label {box.tolist()} has no full-image animal")
        j = int(np.argmin(error))
        used[j] = True
        out[i] = obbs[j]
        out[i, 0] -= x0
        out[i, 1] -= y0
    return out


def enclosing_upright(obb: ArrayLike) -> NDArray[np.float64]:
    """The tightest upright box (x0, y0, x1, y1) around each oriented box."""
    o = np.asarray(obb, float).reshape(-1, 5)
    cos, sin = np.abs(np.cos(o[:, 4])), np.abs(np.sin(o[:, 4]))
    ex = (o[:, 2] * cos + o[:, 3] * sin) / 2
    ey = (o[:, 2] * sin + o[:, 3] * cos) / 2
    return np.column_stack([o[:, 0] - ex, o[:, 1] - ey, o[:, 0] + ex, o[:, 1] + ey])


def obb_quad(obb: ArrayLike) -> list[float]:
    """The four corners as [x1, y1, ..., x4, y4], in order around the box."""
    cx, cy, w, h, a = (float(v) for v in np.asarray(obb, float))
    ux, uy = math.cos(a) * w / 2, math.sin(a) * w / 2
    vx, vy = -math.sin(a) * h / 2, math.cos(a) * h / 2
    return [
        cx - ux - vx,
        cy - uy - vy,
        cx + ux - vx,
        cy + uy - vy,
        cx + ux + vx,
        cy + uy + vy,
        cx - ux + vx,
        cy - uy + vy,
    ]
