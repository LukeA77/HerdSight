"""D-FINE detector exported to ONNX (gpu/train_dfine_baselines_kaggle.py), run with ONNX Runtime.

The exported graph includes D-FINE's post-processor: inputs `images` (N, 3, 640, 640) float32 in
[0, 1] (the training pipeline's ConvertPILImage with scale=True, no mean/std normalisation) and
`orig_target_sizes` (N, 2) int64 as (width, height); outputs `labels`, `boxes` (x0, y0, x1, y1 in
the original tile's pixels) and `scores` for the top queries.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

CATTLE_LABEL = 1


def preprocess(tile: Image.Image, size: int) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Tile -> (1, 3, size, size) float32 in [0, 1], and its (width, height)."""
    w, h = tile.size
    im = tile.convert("RGB")
    if (w, h) != (size, size):
        im = im.resize((size, size), Image.Resampling.BILINEAR)
    x = np.asarray(im, np.float32).transpose(2, 0, 1)[None] / 255.0
    return x, np.array([[w, h]], np.int64)


class DfineOnnxTileDetector:
    """Callable tile detector: PIL tile -> (boxes in tile pixels, scores) for the cattle class."""

    def __init__(
        self, onnx_path: Path | str, min_score: float, size: int = 640, threads: int = 0
    ) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.size = size
        self.min_score = float(min_score)

    def __call__(self, tile: Image.Image) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        x, sizes = preprocess(tile, self.size)
        labels, boxes, scores = self.session.run(None, {"images": x, "orig_target_sizes": sizes})
        keep = (labels[0] == CATTLE_LABEL) & (scores[0] >= self.min_score)
        return boxes[0][keep].astype(float), scores[0][keep].astype(float)
