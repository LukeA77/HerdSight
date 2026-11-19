"""Count errors and calibrated count intervals (brief sections 7 S2/S3 and 9.4).

Intervals use split conformal prediction on normalised count residuals. For each calibration
image (or flight) with predicted count c and true count n, the score is |n - c| / sqrt(c + 1):
the error measured in units of a Poisson-like spread, so one set of scores serves both small and
large herds. For a new prediction c, the interval at level p is c +/- q_p * sqrt(c + 1), rounded
outwards to whole animals and clipped at zero. q_p is the finite-sample conformal quantile.

The scale sqrt(c + 1) is a modelling choice, not a guarantee: coverage within herd-size bands is
checked empirically and reported (`coverage_by_band`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from herdsight.evaluate.conformal import conformal_quantile

DEFAULT_LEVELS = (0.68, 0.90, 0.95)


def count_errors(pred: ArrayLike, true: ArrayLike) -> dict[str, float]:
    """Mean absolute error, root-mean-square error, mean signed error (pred - true), and the
    share of exact counts."""
    p = np.asarray(pred, float).reshape(-1)
    t = np.asarray(true, float).reshape(-1)
    if p.shape != t.shape:
        raise ValueError("pred and true must have the same length")
    if p.size == 0:
        raise ValueError("no counts")
    e = p - t
    return {
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "mean_signed_error": float(np.mean(e)),
        "exact_share": float(np.mean(np.round(p) == t)),
        "n": int(p.size),
    }


def _scale(pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(pred, 0.0) + 1.0)


def fit_count_conformal(
    pred: ArrayLike, true: ArrayLike, levels: Sequence[float] = DEFAULT_LEVELS
) -> dict[float, float]:
    """Conformal quantile q_p of the normalised residual for each level p."""
    p = np.asarray(pred, float).reshape(-1)
    t = np.asarray(true, float).reshape(-1)
    if p.shape != t.shape or p.size == 0:
        raise ValueError("need matching, non-empty calibration counts")
    scores = np.abs(t - p) / _scale(p)
    return {float(lv): conformal_quantile(scores, lv) for lv in levels}


def count_interval(pred: ArrayLike, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Whole-animal interval [lo, hi] around predicted counts for quantile q (inf -> unbounded)."""
    p = np.asarray(pred, float).reshape(-1)
    if not math.isfinite(q):
        return np.zeros_like(p), np.full_like(p, np.inf)
    half = q * _scale(p)
    lo = np.maximum(np.floor(p - half), 0.0)
    hi = np.ceil(p + half)
    return lo, hi


def coverage(pred: ArrayLike, true: ArrayLike, q_by_level: dict[float, float]) -> dict[str, float]:
    """Share of true counts inside the interval, and mean interval width, per level."""
    p = np.asarray(pred, float).reshape(-1)
    t = np.asarray(true, float).reshape(-1)
    out: dict[str, float] = {}
    for lv, q in q_by_level.items():
        lo, hi = count_interval(p, q)
        out[f"coverage_{lv:.2f}"] = float(np.mean((t >= lo) & (t <= hi)))
        out[f"mean_width_{lv:.2f}"] = float(np.mean(hi - lo))
    return out


def coverage_by_band(
    pred: ArrayLike,
    true: ArrayLike,
    q_by_level: dict[float, float],
    edges: Sequence[float] = (0, 1, 5, 20, math.inf),
) -> list[dict[str, float]]:
    """Coverage within bands of the true count (e.g. empty, 1-4, 5-19, 20+)."""
    p = np.asarray(pred, float).reshape(-1)
    t = np.asarray(true, float).reshape(-1)
    rows = []
    for lo_e, hi_e in zip(edges[:-1], edges[1:], strict=True):
        m = (t >= lo_e) & (t < hi_e)
        row: dict[str, float] = {"band_lo": float(lo_e), "band_hi": float(hi_e), "n": int(m.sum())}
        if m.any():
            row.update(coverage(p[m], t[m], q_by_level))
        rows.append(row)
    return rows
