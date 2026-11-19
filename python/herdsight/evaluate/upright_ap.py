"""Standard upright-box AP through pycocotools, for comparison with published work (brief 9.2).

Oriented predictions are first converted to their enclosing upright boxes by the caller
(`herdsight.evaluate.upright_ap.obb_to_upright`). COCO's evaluation is used unchanged except for
the maximum detections per image: COCO's default of 100 would silently drop detections in large
herds, so it is raised (`max_dets`) and the value is recorded with every result.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

DEFAULT_MAX_DETS = 1000


@dataclass(frozen=True)
class ImageBoxes:
    """Upright boxes (x_min, y_min, x_max, y_max) for one image."""

    pred_xyxy: NDArray[np.float64]
    pred_score: NDArray[np.float64]
    true_xyxy: NDArray[np.float64]

    @staticmethod
    def make(pred_xyxy: ArrayLike, pred_score: ArrayLike, true_xyxy: ArrayLike) -> ImageBoxes:
        p = np.asarray(pred_xyxy, float).reshape(-1, 4)
        s = np.asarray(pred_score, float).reshape(-1)
        t = np.asarray(true_xyxy, float).reshape(-1, 4)
        if s.shape[0] != p.shape[0]:
            raise ValueError("one score per predicted box")
        return ImageBoxes(p, s, t)


def obb_to_upright(obb_xywha: ArrayLike) -> NDArray[np.float64]:
    """Enclosing upright boxes of oriented boxes (cx, cy, w, h, angle in radians)."""
    o = np.asarray(obb_xywha, float).reshape(-1, 5)
    c, s = np.abs(np.cos(o[:, 4])), np.abs(np.sin(o[:, 4]))
    half_w = (o[:, 2] * c + o[:, 3] * s) / 2
    half_h = (o[:, 2] * s + o[:, 3] * c) / 2
    return np.column_stack([o[:, 0] - half_w, o[:, 1] - half_h, o[:, 0] + half_w, o[:, 1] + half_h])


def upright_ap(images: Sequence[ImageBoxes], max_dets: int = DEFAULT_MAX_DETS) -> dict[str, float]:
    """COCO bounding-box AP50 and AP50:95 (single class) over the given images."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt: dict = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "animal"}]}
    dets: list[dict] = []
    ann_id = 1
    for img_id, im in enumerate(images, start=1):
        gt["images"].append({"id": img_id})
        for x0, y0, x1, y1 in im.true_xyxy:
            w, h = x1 - x0, y1 - y0
            gt["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [float(x0), float(y0), float(w), float(h)],
                    "area": float(w * h),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
        for (x0, y0, x1, y1), sc in zip(im.pred_xyxy, im.pred_score, strict=True):
            dets.append(
                {
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                    "score": float(sc),
                }
            )
    n_true = ann_id - 1
    if n_true == 0:
        return {"ap50": float("nan"), "ap50_95": float("nan"), "n_true": 0, "max_dets": max_dets}
    if not dets:
        return {"ap50": 0.0, "ap50_95": 0.0, "n_true": n_true, "max_dets": max_dets}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dets)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.params.maxDets = [1, 10, max_dets]
        ev.evaluate()
        ev.accumulate()
    # Read AP from the precision table rather than ev.summarize(): summarize() hard-codes
    # maxDets=100 for AP50:95 and returns -1 for any other limit.
    # precision has shape [iou thresholds, recall points, classes, area ranges, max dets].
    prec = ev.eval["precision"][:, :, 0, 0, -1]  # one class, all areas, the largest limit

    def _mean(p: np.ndarray) -> float:
        valid = p[p > -1]
        return float(valid.mean()) if valid.size else float("nan")

    return {
        "ap50": _mean(prec[0]),
        "ap50_95": _mean(prec),
        "n_true": n_true,
        "max_dets": max_dets,
    }


def iou_matrix(a_xyxy: ArrayLike, b_xyxy: ArrayLike) -> NDArray[np.float64]:
    """Intersection over union between every box in a (rows) and in b (columns)."""
    a = np.asarray(a_xyxy, float).reshape(-1, 4)
    b = np.asarray(b_xyxy, float).reshape(-1, 4)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def match_iou(pred_xyxy: ArrayLike, true_xyxy: ArrayLike, threshold: float = 0.5) -> int:
    """Number of one-to-one matches with IoU >= threshold (optimal assignment)."""
    from scipy.optimize import linear_sum_assignment

    m = iou_matrix(pred_xyxy, true_xyxy)
    if m.size == 0:
        return 0
    w = np.where(m >= threshold, m, 0.0)
    r, c = linear_sum_assignment(w, maximize=True)
    return int(np.sum(w[r, c] >= threshold))
