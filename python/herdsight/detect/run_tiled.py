"""Run any tile detector over a full-resolution still and merge by centre ownership."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from herdsight.detect.tiling import keep_owned, tile_grid

#: A tile detector: PIL tile -> (boxes (N, 4) in tile pixels, scores (N,)).
TileDetector = Callable[[Image.Image], tuple[NDArray[np.float64], NDArray[np.float64]]]


def detect_full_image(
    image: Image.Image, detector: TileDetector, tile: int, overlap: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Boxes (full-image pixels) and scores for the whole still. Each detection is kept only by
    the tile that owns its centre, so the overlap never double-counts an animal."""
    w, h = image.size
    all_boxes, all_scores = [], []
    for t in tile_grid(w, h, tile, overlap):
        boxes, scores = detector(image.crop((t.x0, t.y0, t.x1, t.y1)))
        boxes = np.asarray(boxes, float).reshape(-1, 4) + [t.x0, t.y0, t.x0, t.y0]
        keep = keep_owned(t, boxes)
        all_boxes.append(boxes[keep])
        all_scores.append(np.asarray(scores, float).reshape(-1)[keep])
    return np.concatenate(all_boxes), np.concatenate(all_scores)
