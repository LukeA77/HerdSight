"""Orientation-independent detection scores from animal centres (brief section 9.1).

Predictions may be oriented boxes while the labels are upright boxes, so box overlap (IoU) would
punish correct rotated predictions. Instead, predicted and true *centres* are compared with a
similarity score in [0, 1], analogous to COCO's Object Keypoint Similarity (OKS):

    s = exp(-d^2 / (2 * (kappa * sigma)^2))

where d is the centre distance in pixels and sigma is the true animal's size (by default the short
side of its upright box). kappa is set from measured label noise and recorded in configs.

Two uses, two matching rules:

* `match_hungarian`: one-to-one optimal matching of the kept detections, for precision, recall and
  counts at an operating point.
* `average_precision`: COCO's procedure (predictions in descending score; each takes the unmatched
  truth with the highest similarity above the threshold), as COCO uses for keypoint AP. Greedy
  matching in score order is what makes AP well defined across score thresholds; using the same
  procedure keeps the numbers comparable in meaning with COCO-style AP.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment

#: COCO's similarity thresholds: 0.50, 0.55, ..., 0.95.
COCO_THRESHOLDS: NDArray[np.float64] = np.round(np.arange(0.50, 0.951, 0.05), 2)
#: COCO's 101 recall points for interpolated precision.
RECALL_POINTS: NDArray[np.float64] = np.linspace(0.0, 1.0, 101)


def short_side(boxes_xyxy: ArrayLike) -> NDArray[np.float64]:
    """Short side (pixels) of upright boxes given as (x_min, y_min, x_max, y_max)."""
    b = np.asarray(boxes_xyxy, float).reshape(-1, 4)
    return np.minimum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])


def box_centres(boxes_xyxy: ArrayLike) -> NDArray[np.float64]:
    """Centres of upright boxes (x_min, y_min, x_max, y_max), shape (N, 2)."""
    b = np.asarray(boxes_xyxy, float).reshape(-1, 4)
    return np.column_stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2])


def similarity_matrix(
    pred_xy: ArrayLike, true_xy: ArrayLike, true_sigma: ArrayLike, kappa: float
) -> NDArray[np.float64]:
    """Centre similarity between every prediction (rows) and every truth (columns)."""
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    p = np.asarray(pred_xy, float).reshape(-1, 2)
    t = np.asarray(true_xy, float).reshape(-1, 2)
    sigma = np.asarray(true_sigma, float).reshape(-1)
    if sigma.shape[0] != t.shape[0]:
        raise ValueError("one sigma per true centre")
    if np.any(sigma <= 0):
        raise ValueError("true sizes must be positive")
    d2 = ((p[:, None, :] - t[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-d2 / (2.0 * (kappa * sigma[None, :]) ** 2))


@dataclass(frozen=True)
class Matches:
    """Result of one-to-one matching at a similarity threshold."""

    pairs: NDArray[np.int64]  # (K, 2): prediction index, truth index
    similarity: NDArray[np.float64]  # (K,)
    n_pred: int
    n_true: int

    @property
    def tp(self) -> int:
        return int(self.pairs.shape[0])

    @property
    def fp(self) -> int:
        return self.n_pred - self.tp

    @property
    def fn(self) -> int:
        return self.n_true - self.tp


def match_hungarian(
    pred_xy: ArrayLike,
    true_xy: ArrayLike,
    true_sigma: ArrayLike,
    kappa: float,
    threshold: float = 0.5,
) -> Matches:
    """Optimal one-to-one matching maximising total similarity; pairs below `threshold` are
    discarded (they count as one false positive and one false negative)."""
    s = similarity_matrix(pred_xy, true_xy, true_sigma, kappa)
    n_pred, n_true = s.shape
    if n_pred == 0 or n_true == 0:
        return Matches(np.zeros((0, 2), np.int64), np.zeros(0), n_pred, n_true)
    # Pairs below the threshold may not be matched: give them zero weight so the optimiser never
    # trades a valid pair for an invalid one.
    w = np.where(s >= threshold, s, 0.0)
    rows, cols = linear_sum_assignment(w, maximize=True)
    keep = w[rows, cols] >= threshold
    pairs = np.column_stack([rows[keep], cols[keep]]).astype(np.int64)
    return Matches(pairs, s[rows[keep], cols[keep]], n_pred, n_true)


@dataclass(frozen=True)
class ImageDetections:
    """Predictions and truth for one image."""

    pred_xy: NDArray[np.float64]  # (P, 2)
    pred_score: NDArray[np.float64]  # (P,)
    true_xy: NDArray[np.float64]  # (T, 2)
    true_sigma: NDArray[np.float64]  # (T,)

    @staticmethod
    def make(
        pred_xy: ArrayLike, pred_score: ArrayLike, true_xy: ArrayLike, true_sigma: ArrayLike
    ) -> ImageDetections:
        p = np.asarray(pred_xy, float).reshape(-1, 2)
        sc = np.asarray(pred_score, float).reshape(-1)
        t = np.asarray(true_xy, float).reshape(-1, 2)
        sg = np.asarray(true_sigma, float).reshape(-1)
        if sc.shape[0] != p.shape[0] or sg.shape[0] != t.shape[0]:
            raise ValueError("scores must match predictions and sizes must match truths")
        return ImageDetections(p, sc, t, sg)


def _greedy_tp(img: ImageDetections, kappa: float, threshold: float) -> NDArray[np.bool_]:
    """COCO matching for one image: True where the (score-sorted) prediction is a true positive.
    Returned in the image's original prediction order."""
    n_pred = img.pred_xy.shape[0]
    tp = np.zeros(n_pred, bool)
    if n_pred == 0 or img.true_xy.shape[0] == 0:
        return tp
    s = similarity_matrix(img.pred_xy, img.true_xy, img.true_sigma, kappa)
    taken = np.zeros(img.true_xy.shape[0], bool)
    for i in np.argsort(-img.pred_score, kind="stable"):
        cand = np.where(taken, -1.0, s[i])
        j = int(np.argmax(cand))
        if cand[j] >= threshold:
            taken[j] = True
            tp[i] = True
    return tp


def match_table(
    images: Sequence[ImageDetections], kappa: float, thresholds: ArrayLike = COCO_THRESHOLDS
) -> list[NDArray[np.bool_]]:
    """Per image, a (P, K) table: is prediction p a true positive at similarity threshold k.

    COCO matching never crosses images, so the table can be computed once and reused when images
    are resampled (bootstrap intervals for AP)."""
    thr = np.asarray(thresholds, float).reshape(-1)
    return [
        np.column_stack([_greedy_tp(img, kappa, t) for t in thr])
        if img.pred_xy.shape[0]
        else np.zeros((0, thr.shape[0]), bool)
        for img in images
    ]


def _stack(parts: Sequence[NDArray[np.bool_]], k: int) -> NDArray[np.bool_]:
    return np.concatenate(parts, axis=0) if len(parts) else np.zeros((0, k), bool)


def _ap_per_threshold(scores: NDArray, tp: NDArray[np.bool_], n_true: int) -> NDArray[np.float64]:
    """101-point interpolated AP per column of tp (one column per similarity threshold)."""
    ap = np.full(tp.shape[1], np.nan)
    if n_true == 0:
        return ap
    tp = tp[np.argsort(-scores, kind="stable")]
    for k in range(tp.shape[1]):
        ctp = np.cumsum(tp[:, k])
        cfp = np.cumsum(~tp[:, k])
        recall = ctp / n_true
        precision = ctp / np.maximum(ctp + cfp, 1)
        # Make precision monotonically non-increasing from the right (COCO's envelope).
        precision = np.maximum.accumulate(precision[::-1])[::-1] if precision.size else precision
        idx = np.searchsorted(recall, RECALL_POINTS, side="left")
        sampled = np.array([precision[i] if i < precision.size else 0.0 for i in idx], dtype=float)
        ap[k] = float(sampled.mean())
    return ap


def ap_mean_subset(
    images: Sequence[ImageDetections], table: Sequence[NDArray[np.bool_]], idx: ArrayLike
) -> float:
    """Mean AP over thresholds for the images at idx (repeats allowed), from a match_table."""
    idx = np.asarray(idx, int).reshape(-1)
    n_true = int(sum(images[i].true_xy.shape[0] for i in idx))
    k = table[0].shape[1] if len(table) else 0
    scores = np.concatenate([images[i].pred_score for i in idx]) if idx.size else np.zeros(0, float)
    ap = _ap_per_threshold(scores, _stack([table[i] for i in idx], k), n_true)
    return float(np.nanmean(ap)) if np.isfinite(ap).any() else float("nan")


def average_precision(
    images: Sequence[ImageDetections],
    kappa: float,
    thresholds: ArrayLike = COCO_THRESHOLDS,
) -> dict[str, object]:
    """COCO-style 101-point interpolated AP at each similarity threshold, and their mean.

    Images with no truth still count: their predictions are false positives. If there is no truth
    at all, AP is undefined and returned as NaN.
    """
    thr = np.asarray(thresholds, float).reshape(-1)
    table = match_table(images, kappa, thr)
    n_true = sum(img.true_xy.shape[0] for img in images)
    scores = np.concatenate([img.pred_score for img in images]) if images else np.zeros(0)
    ap = _ap_per_threshold(scores, _stack(table, thr.shape[0]), n_true)
    return {
        "thresholds": thr.tolist(),
        "ap": ap.tolist(),
        "ap_mean": float(np.nanmean(ap)) if np.isfinite(ap).any() else float("nan"),
        "n_true": int(n_true),
        "n_pred": int(scores.shape[0]),
    }
