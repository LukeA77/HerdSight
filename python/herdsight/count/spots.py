"""Choose where to take a second look after the survey (uncertain herd spots).

Specification: _drafts/second-look-spec.md, section 1. The survey's detections, in ground metres,
are chained into groups (animals within `chain_m` of another animal of the group, the herd chain
used for S7). A group of at least `min_group` confident animals is uncertain when any of these
hold: a box at least `big_box_ratio` times the flight's median box area (likely two animals in one
box); at least `min_faint` faint detections (score from `faint_min` up to the operating threshold)
within `chain_m` of the group; at least `touching_share` of its animals touching another animal
(boxes overlapping or less than `touch_gap_m` apart). Spots are ranked by expected hidden animals
(faint detections plus the extra animals implied by oversized boxes) and spots whose watch
footprints overlap by more than `merge_overlap` are merged. At most `max_spots` are returned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class Spot:
    centre: np.ndarray  # east, north (m)
    animals: int  # confident detections in the group
    hidden_estimate: float  # faint detections + extra animals from oversized boxes
    reasons: tuple[str, ...]


def _gaps(boxes: np.ndarray) -> np.ndarray:
    """Pairwise gap between boxes (x0, y0, x1, y1) in metres; 0 where they overlap."""
    dx = np.maximum(
        0.0,
        np.maximum(boxes[:, None, 0], boxes[None, :, 0])
        - np.minimum(boxes[:, None, 2], boxes[None, :, 2]),
    )
    dy = np.maximum(
        0.0,
        np.maximum(boxes[:, None, 1], boxes[None, :, 1])
        - np.minimum(boxes[:, None, 3], boxes[None, :, 3]),
    )
    return np.hypot(dx, dy)


def _groups(centres: np.ndarray, chain_m: float) -> np.ndarray:
    n = len(centres)
    d = np.linalg.norm(centres[:, None] - centres[None], axis=2)
    i, j = np.nonzero((d <= chain_m) & ~np.eye(n, dtype=bool))
    graph = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def _footprint_overlap(a: np.ndarray, b: np.ndarray, size: tuple[float, float]) -> float:
    """Share of one watch footprint (width, height, centred on a and b) covered by the other."""
    w, h = size
    ox = max(0.0, w - abs(a[0] - b[0]))
    oy = max(0.0, h - abs(a[1] - b[1]))
    return ox * oy / (w * h)


def choose_spots(
    boxes: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    faint_min: float = 0.1,
    chain_m: float = 10.0,
    min_group: int = 3,
    big_box_ratio: float = 1.6,
    min_faint: int = 2,
    touching_share: float = 0.3,
    touch_gap_m: float = 0.2,
    footprint_m: tuple[float, float] = (60.0, 45.0),
    merge_overlap: float = 0.3,
    max_spots: int = 5,
) -> list[Spot]:
    """Uncertain spots from one flight's ground boxes (x0, y0, x1, y1 metres) and scores."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    scores = np.asarray(scores, float)
    if len(boxes) != len(scores):
        raise ValueError("one score per box")
    sure = scores >= threshold
    faint = (scores >= faint_min) & ~sure
    b, fb = boxes[sure], boxes[faint]
    if len(b) < min_group:
        return []
    centres = (b[:, :2] + b[:, 2:]) / 2
    faint_c = (fb[:, :2] + fb[:, 2:]) / 2
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    median_area = float(np.median(area))
    touching = (_gaps(b) < touch_gap_m) & ~np.eye(len(b), dtype=bool)
    labels = _groups(centres, chain_m)
    spots = []
    for g in np.unique(labels):
        idx = np.nonzero(labels == g)[0]
        if len(idx) < min_group:
            continue
        big = area[idx] >= big_box_ratio * median_area
        extra = float(np.sum(np.round(area[idx][big] / median_area) - 1))
        near = 0
        if len(faint_c):
            d = np.linalg.norm(faint_c[:, None] - centres[idx][None], axis=2)
            near = int(np.sum(d.min(axis=1) <= chain_m))
        touch = float(np.mean(touching[np.ix_(idx, np.arange(len(b)))].any(axis=1)))
        reasons = []
        if big.any():
            reasons.append("oversized box")
        if near >= min_faint:
            reasons.append("faint detections")
        if touch >= touching_share:
            reasons.append("touching animals")
        if reasons:
            spots.append(Spot(centres[idx].mean(axis=0), len(idx), near + extra, tuple(reasons)))
    spots.sort(key=lambda s: -s.hidden_estimate)
    merged: list[Spot] = []
    for s in spots:
        for k, m in enumerate(merged):
            if _footprint_overlap(s.centre, m.centre, footprint_m) > merge_overlap:
                n = m.animals + s.animals
                merged[k] = Spot(
                    (m.centre * m.animals + s.centre * s.animals) / n,
                    n,
                    m.hidden_estimate + s.hidden_estimate,
                    tuple(dict.fromkeys(m.reasons + s.reasons)),
                )
                break
        else:
            merged.append(s)
    merged.sort(key=lambda s: -s.hidden_estimate)
    return merged[:max_spots]
