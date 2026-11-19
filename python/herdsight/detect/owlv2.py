"""Zero-shot baseline: OWLv2 open-vocabulary detection with no cattle training (Phase 1).

Model: google/owlv2-base-patch16-ensemble (Apache-2.0), through Hugging Face transformers
(Apache-2.0). The text queries and the minimum score kept are fixed in configs/baselines.yaml;
nothing is tuned. Torch and transformers are imported only when this module is used, so the core
package does not depend on them.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from PIL import Image


class Owlv2TileDetector:
    """Callable tile detector: PIL tile -> (boxes in tile pixels, scores)."""

    def __init__(
        self,
        model_id: str,
        revision: str | None,
        queries: Sequence[str],
        min_score: float,
        threads: int | None = None,
    ) -> None:
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        if threads:
            torch.set_num_threads(threads)
        self._torch = torch
        self.processor = Owlv2Processor.from_pretrained(model_id, revision=revision)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id, revision=revision).eval()
        self.queries = [list(queries)]
        self.min_score = float(min_score)

    def __call__(self, tile: Image.Image) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        torch = self._torch
        w, h = tile.size
        with torch.inference_mode():
            inputs = self.processor(
                text=self.queries, images=tile.convert("RGB"), return_tensors="pt"
            )
            out = self.model(**inputs)
        # OWLv2 pads the image to a square at the bottom and right, so boxes are scaled to the
        # padded side and then lie in the original tile's pixel frame.
        side = max(w, h)
        res = self.processor.post_process_grounded_object_detection(
            out, threshold=self.min_score, target_sizes=[(side, side)]
        )[0]
        boxes = res["boxes"].cpu().numpy().astype(float).reshape(-1, 4)
        scores = res["scores"].cpu().numpy().astype(float).reshape(-1)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        return boxes, scores
