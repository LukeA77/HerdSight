"""Oriented training labels from upright boxes with the Segment Anything Model (brief 8.4).

For each upright box, a square crop around it (at native resolution, padded by `context` times
the box size) is segmented by SAM with the box as the prompt. The best-scoring mask is reduced to
its tightest rotated rectangle (`herdsight.labels.obb.mask_to_obb`). Each label carries SAM's
predicted mask quality and flags for review:

* `no_mask`        segmentation failed (fewer than a handful of pixels);
* `escapes_box`    the oriented box's upright hull extends well beyond the original box, so the
                   mask probably took in a neighbour or shadow;
* `low_fill`       the mask fills little of the original box, so the animal was probably missed.

Flagged labels are checked by eye (Phase 2); none is silently corrected.
Model: facebook/sam-vit-base (Apache-2.0) via Hugging Face transformers (Apache-2.0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from herdsight.evaluate.upright_ap import iou_matrix, obb_to_upright
from herdsight.labels.obb import mask_to_obb


@dataclass(frozen=True)
class ObbLabel:
    obb: NDArray[np.float64] | None  # (cx, cy, w, h, angle) in full-image pixels
    sam_score: float
    hull_iou_with_box: float  # IoU of the oriented box's upright hull with the original box
    mask_fill: float  # mask pixels / original box area
    flags: tuple[str, ...]


class SamObbLabeller:
    def __init__(
        self,
        model_id: str,
        revision: str | None,
        context: float = 0.5,
        escape_iou: float = 0.6,
        min_fill: float = 0.25,
        threads: int | None = None,
        device: str = "cpu",
    ) -> None:
        import torch
        from transformers import SamModel, SamProcessor

        if threads:
            torch.set_num_threads(threads)
        self._torch = torch
        self.processor = SamProcessor.from_pretrained(model_id, revision=revision)
        self.device = device
        self.model = SamModel.from_pretrained(model_id, revision=revision).to(device).eval()
        self.context, self.escape_iou, self.min_fill = context, escape_iou, min_fill

    def label(self, image: Image.Image, box_xyxy: NDArray[np.float64]) -> ObbLabel:
        torch = self._torch
        x0, y0, x1, y1 = (float(v) for v in box_xyxy)
        pad = self.context * max(x1 - x0, y1 - y0)
        cx0, cy0 = max(0, int(x0 - pad)), max(0, int(y0 - pad))
        cx1, cy1 = min(image.width, int(x1 + pad) + 1), min(image.height, int(y1 + pad) + 1)
        crop = image.crop((cx0, cy0, cx1, cy1)).convert("RGB")
        prompt = [[[x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0]]]
        with torch.inference_mode():
            inputs = self.processor(crop, input_boxes=prompt, return_tensors="pt").to(self.device)
            out = self.model(**inputs, multimask_output=True)
        masks = self.processor.image_processor.post_process_masks(
            out.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0][0]  # (3, H, W) for the one box
        scores = out.iou_scores[0, 0].cpu().numpy()
        k = int(np.argmax(scores))
        mask = masks[k].cpu().numpy()
        obb = mask_to_obb(mask)
        box_area = max((x1 - x0) * (y1 - y0), 1e-9)
        fill = float(
            mask[int(y0 - cy0) : int(y1 - cy0) + 1, int(x0 - cx0) : int(x1 - cx0) + 1].sum()
        )
        fill /= box_area
        if obb is None:
            return ObbLabel(None, float(scores[k]), 0.0, fill, ("no_mask",))
        obb = obb + np.array([cx0, cy0, 0, 0, 0], float)
        hull_iou = float(iou_matrix(obb_to_upright([obb]), [box_xyxy])[0, 0])
        flags = []
        if hull_iou < self.escape_iou:
            flags.append("escapes_box")
        if fill < self.min_fill:
            flags.append("low_fill")
        return ObbLabel(obb, float(scores[k]), hull_iou, fill, tuple(flags))
