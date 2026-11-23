"""From raw SAM results to the oriented training labels (Phase 2, step C6).

* A flagged or failed SAM result falls back to an angle-0 box of the upright size. It is never
  dropped, because that would leave an unlabelled animal among the touching ones.
* If SAM's centres do not meet the frozen centre-recall rule, every oriented box keeps the upright
  box's centre and SAM supplies only angle and size.
* `select_for_review` picks the labels checked by eye, stratified by fold and by flag. It uses a
  fixed seed and no model result, so the sample cannot be chosen to look good.

These labels are for training only and are never used to score a model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from herdsight.labels.obb import normalise_obb

MIN_PER_CELL = 5


def final_obb(
    box_xyxy: list[float], label: dict[str, Any], keep_upright_centre: bool
) -> NDArray[np.float64]:
    """The (cx, cy, w, h, angle) used for training."""
    x0, y0, x1, y1 = box_xyxy
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if label["obb"] is None or label["flags"]:
        return normalise_obb(cx, cy, x1 - x0, y1 - y0, 0.0)
    ox, oy, w, h, angle = label["obb"]
    if keep_upright_centre:
        ox, oy = cx, cy
    return normalise_obb(ox, oy, w, h, angle)


def _allocate(sizes: dict[tuple[str, bool], int], n: int) -> dict[tuple[str, bool], int]:
    """Share `n` across cells in proportion to size, with a floor for each non-empty cell."""
    cells = [c for c, s in sizes.items() if s > 0]
    take = {c: min(MIN_PER_CELL, sizes[c]) for c in cells}
    left = n - sum(take.values())
    if left < 0:
        raise ValueError(f"{n} labels cannot cover {len(cells)} cells")
    while left > 0:
        open_cells = [c for c in cells if take[c] < sizes[c]]
        if not open_cells:
            break
        c = max(open_cells, key=lambda x: (sizes[x] / (take[x] + 1), str(x)))
        take[c] += 1
        left -= 1
    return take


def select_for_review(
    rows: list[dict[str, Any]], n: int = 100, seed: int = 0
) -> list[tuple[str, int]]:
    """(image, box index) of `n` labels, stratified by fold and by whether SAM flagged them."""
    by_cell: dict[tuple[str, bool], list[tuple[str, int]]] = {}
    for row in sorted(rows, key=lambda r: r["image"]):
        for k, label in enumerate(row["labels"]):
            flagged = label["obb"] is None or bool(label["flags"])
            by_cell.setdefault((row["fold"], flagged), []).append((row["image"], k))
    take = _allocate({c: len(v) for c, v in by_cell.items()}, n)
    rng = np.random.default_rng(seed)
    out: list[tuple[str, int]] = []
    for cell in sorted(by_cell, key=str):
        members = by_cell[cell]
        for i in rng.choice(len(members), take[cell], replace=False):
            out.append(members[int(i)])
    order = rng.permutation(len(out))  # shown in a mixed order, so flagged labels do not cluster
    return [out[int(i)] for i in order]
