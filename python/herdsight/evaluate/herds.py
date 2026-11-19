"""Herd recall for onboard herd detection (S7, D-28).

A herd is a group of at least `min_animals` animals in which each is within `link_m` metres of
another (single linkage on ground positions). Two recall measures are reported:

* found: at least one kept detection has its centre inside the box of one of the herd's animals
  (D-28's definition: enough for the drone to stop and look);
* grouped: the kept detections themselves form a predicted herd (same rule), and that predicted
  herd has a detection inside at least one of the true herd's boxes. This is what herd mode would
  actually react to if it triggers only on groups.

False triggers count predicted herds that touch no true herd.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist


def groups(centres_m: ArrayLike, link_m: float, min_size: int) -> list[NDArray[np.int64]]:
    """Index arrays of single-linkage groups (distance < link_m) with at least min_size members."""
    c = np.asarray(centres_m, float).reshape(-1, 2)
    if len(c) < min_size or len(c) == 0:
        return []
    _, lab = connected_components(cdist(c, c) < link_m, directed=False)
    out = [np.nonzero(lab == g)[0] for g in range(lab.max() + 1)]
    return [g for g in out if len(g) >= min_size]


def centres_inside(boxes_xyxy: ArrayLike, points: ArrayLike) -> NDArray[np.bool_]:
    """(boxes, points) matrix: is point j inside box i (edges included)?"""
    b = np.asarray(boxes_xyxy, float).reshape(-1, 4)
    p = np.asarray(points, float).reshape(-1, 2)
    return (
        (p[None, :, 0] >= b[:, None, 0])
        & (p[None, :, 0] <= b[:, None, 2])
        & (p[None, :, 1] >= b[:, None, 1])
        & (p[None, :, 1] <= b[:, None, 3])
    )


@dataclass(frozen=True)
class ImageHerds:
    true_herds: int
    found: int
    grouped: int
    predicted_herds: int
    false_triggers: int


def score_image(
    true_boxes: ArrayLike,
    pred_boxes: ArrayLike,
    gsd_m: float,
    link_m: float,
    min_animals: int,
) -> ImageHerds:
    """Herd outcomes for one still. Boxes are in full-image pixels; gsd_m is metres per pixel."""
    tb = np.asarray(true_boxes, float).reshape(-1, 4)
    pb = np.asarray(pred_boxes, float).reshape(-1, 4)
    tc = np.column_stack([(tb[:, 0] + tb[:, 2]) / 2, (tb[:, 1] + tb[:, 3]) / 2])
    pc = np.column_stack([(pb[:, 0] + pb[:, 2]) / 2, (pb[:, 1] + pb[:, 3]) / 2])
    true_g = groups(tc * gsd_m, link_m, min_animals)
    pred_g = groups(pc * gsd_m, link_m, min_animals)
    inside = centres_inside(tb, pc)  # (true animals, predictions)
    found = sum(bool(inside[g].any()) for g in true_g)
    grouped = sum(any(bool(inside[np.ix_(g, pg)].any()) for pg in pred_g) for g in true_g)
    touches_any = [any(bool(inside[np.ix_(g, pg)].any()) for g in true_g) for pg in pred_g]
    return ImageHerds(
        true_herds=len(true_g),
        found=found,
        grouped=grouped,
        predicted_herds=len(pred_g),
        false_triggers=sum(not t for t in touches_any),
    )
