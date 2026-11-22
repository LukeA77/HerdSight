"""Flight-count ranges by split conformal prediction (Phase 3 plan, step 6; D-9).

The error of a flight count is mostly proportional to the herd, so ranges are built on the log
ratio e = log(true / count). From calibration flights:

* the centre m is the median of e (a bias correction: the count times exp(m) is the corrected
  count, used only if it helps out of fold);
* the conformity score is |e - m|, and the range at level L is count * exp(m -+ q_L), where q_L is
  the ceil((n + 1) L)-th smallest score of the n calibration flights. With too few flights for a
  level (ceil((n + 1) L) > n) the range is unbounded, and is reported as such rather than
  shrunk.

Under exchangeability of flights this gives at least L coverage; with 13 or 14 calibration
flights per fold, 0.68 and 0.90 are finite and 0.95 is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class Calibration:
    centre: float  # median log(true / count)
    q: dict[float, float]  # level -> half-width in log space (inf when too few flights)
    n: int

    def corrected(self, count: float) -> float:
        return float(count * math.exp(self.centre))

    def interval(self, count: float, level: float) -> tuple[float, float]:
        q = self.q[level]
        if not math.isfinite(q) or count <= 0:
            return (0.0, math.inf)
        return (
            float(count * math.exp(self.centre - q)),
            float(count * math.exp(self.centre + q)),
        )


def fit(counts: ArrayLike, truths: ArrayLike, levels: tuple[float, ...]) -> Calibration:
    c, t = np.asarray(counts, float), np.asarray(truths, float)
    ok = (c > 0) & (t > 0)
    e = np.log(t[ok] / c[ok])
    n = int(e.size)
    m = float(np.median(e)) if n else 0.0
    s = np.sort(np.abs(e - m))
    q = {}
    for lv in levels:
        k = math.ceil((n + 1) * lv)
        q[lv] = float(s[k - 1]) if 1 <= k <= n else math.inf
    return Calibration(m, q, n)


def inside(truth: float, lo: float, hi: float) -> bool:
    """Whole-animal counts: a range [lo, hi] contains the truth if floor(lo) <= truth <=
    ceil(hi), so a range of 141.3 to 144.6 reads "141 to 145"."""
    return math.floor(lo) <= truth <= (math.ceil(hi) if math.isfinite(hi) else math.inf)
