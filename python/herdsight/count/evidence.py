"""Evidence that two detections in overlapping stills are one animal (Phase 3 plan, step 4).

For every pair of overlapping stills (i, j) and every detection a in i and b in j closer than a
wide gate, a candidate pair is made with:

* distance d in metres on the ground: between a and b's position mapped into still i by the
  registration (pixels times still i's ground sampling distance), or, where registration failed,
  between their geotag ground positions;
* time between the two stills, dt;
* the movement allowance: how far an animal may graze in dt, from the growth of the squared
  position difference of linked pairs with time, d^2 = floor^2 + 4 D dt (the herd-motion model
  of scripts/herd_motion_stats.py, refitted per fold; only the growth D is used);
* competition: whether a and b are each other's closest candidate, and by how much the next
  candidate of either is further away (in herds a pair is only convincing if no neighbour fits
  nearly as well);
* size: the difference in log box length (metres), and the lower of the two detection scores.

Two feature sets are built from the same pairs. "fixed" scales distance by the Phase 1 gate (half
the median animal length, or 3 m for geotags; D-30); "movement" by the same gate widened by the
grazing allowance, scale^2 = gate^2 + 4 D dt. The two differ only in the time term, which is what
H24 compares. A logistic model (`fit_logistic`) turns the features into the probability of
one animal; its log odds are the costs of the grouping step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment, minimize

FEATURES = ("dist2", "margin", "mutual", "size", "low_score", "geotag")
MARGIN_CAP = 5.0


@dataclass(frozen=True)
class MotionModel:
    """Expected squared position difference of one animal between two stills dt apart."""

    floor_m: float
    diffusion_m2_s: float

    def allowance(self, dt_s: NDArray) -> NDArray:
        return np.sqrt(self.floor_m**2 + 4.0 * self.diffusion_m2_s * np.abs(dt_s))

    @staticmethod
    def fit(d_m: NDArray, dt_s: NDArray) -> MotionModel:
        """Least squares of d^2 = floor^2 + 4 D dt, both terms kept non-negative."""
        d2, dt = np.asarray(d_m, float) ** 2, np.abs(np.asarray(dt_s, float))
        a = np.column_stack([np.ones_like(dt), 4.0 * dt])
        coef, *_ = np.linalg.lstsq(a, d2, rcond=None)
        if coef[1] < 0:  # no growth with time: floor only
            coef = np.array([d2.mean(), 0.0])
        if coef[0] < 0:  # through the origin
            coef = np.array([0.0, float(d2 @ (4 * dt) / max((4 * dt) @ (4 * dt), 1e-12))])
        return MotionModel(float(np.sqrt(coef[0])), float(coef[1]))


@dataclass
class Still:
    time_s: float
    gsd_m: float  # metres per pixel at the ground
    centres_px: NDArray  # (n, 2)
    ground_m: NDArray  # (n, 2), geotag projection, north-east metres
    length_m: NDArray  # (n,), long side of each box on the ground
    score: NDArray  # (n,), detection score (1 for labels)


@dataclass
class Overlap:
    i: int
    j: int
    h_ab: NDArray | None  # maps still j's pixels to still i's; None when registration failed


@dataclass
class Candidates:
    """One row per candidate pair (still i, detection a) - (still j, detection b)."""

    i: NDArray
    a: NDArray
    j: NDArray
    b: NDArray
    d_m: NDArray
    dt_s: NDArray
    geotag: NDArray  # bool
    fixed_gate_m: NDArray
    size: NDArray
    low_score: NDArray
    extra: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.i)


def _apply_h(h: NDArray, pts: NDArray) -> NDArray:
    p = np.column_stack([pts, np.ones(len(pts))]) @ np.asarray(h, float).T
    return p[:, :2] / p[:, 2:3]


def candidates(
    stills: list[Still],
    overlaps: list[Overlap],
    gate_m: float = 4.0,
    geotag_gate_m: float = 10.0,
    fixed_geotag_gate_m: float = 3.0,
    fixed_gate_fraction: float = 0.5,
) -> Candidates:
    cols: dict[str, list] = {k: [] for k in ("i", "a", "j", "b", "d", "dt", "geo", "fix")}
    for o in overlaps:
        si, sj = stills[o.i], stills[o.j]
        if not len(si.centres_px) or not len(sj.centres_px):
            continue
        if o.h_ab is not None:
            d = (
                np.linalg.norm(
                    si.centres_px[:, None] - _apply_h(o.h_ab, sj.centres_px)[None], axis=2
                )
                * si.gsd_m
            )
            gate, geo = gate_m, False
            fix = fixed_gate_fraction * float(np.median(np.r_[si.length_m, sj.length_m]))
        else:
            d = np.linalg.norm(si.ground_m[:, None] - sj.ground_m[None], axis=2)
            gate, geo, fix = geotag_gate_m, True, fixed_geotag_gate_m
        a, b = np.nonzero(d <= gate)
        cols["i"].append(np.full(len(a), o.i))
        cols["a"].append(a)
        cols["j"].append(np.full(len(a), o.j))
        cols["b"].append(b)
        cols["d"].append(d[a, b])
        cols["dt"].append(np.full(len(a), abs(sj.time_s - si.time_s)))
        cols["geo"].append(np.full(len(a), geo))
        cols["fix"].append(np.full(len(a), fix))
    cat = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}
    i, a, j, b = (cat[k].astype(int) for k in ("i", "a", "j", "b"))
    la = np.array([stills[x].length_m[y] for x, y in zip(i, a, strict=True)])
    lb = np.array([stills[x].length_m[y] for x, y in zip(j, b, strict=True)])
    sa = np.array([stills[x].score[y] for x, y in zip(i, a, strict=True)])
    sb = np.array([stills[x].score[y] for x, y in zip(j, b, strict=True)])
    return Candidates(
        i=i,
        a=a,
        j=j,
        b=b,
        d_m=cat["d"].astype(float),
        dt_s=cat["dt"].astype(float),
        geotag=cat["geo"].astype(bool),
        fixed_gate_m=cat["fix"].astype(float),
        size=np.abs(np.log(np.maximum(la, 1e-6) / np.maximum(lb, 1e-6))) if len(i) else la,
        low_score=1.0 - np.minimum(sa, sb) if len(i) else sa,
    )


def _competition(c: Candidates, z: NDArray) -> tuple[NDArray, NDArray]:
    """Per candidate: margin to the next candidate of a (towards still j) or of b (towards
    still i), and whether the pair is each other's closest."""
    margin = np.full(len(c), MARGIN_CAP)
    mutual = np.zeros(len(c))
    best_a: dict[tuple, float] = {}
    best_b: dict[tuple, float] = {}
    second_a: dict[tuple, float] = {}
    second_b: dict[tuple, float] = {}
    keys_a = list(zip(c.i, c.a, c.j, strict=True))
    keys_b = list(zip(c.j, c.b, c.i, strict=True))
    for best, second, keys in ((best_a, second_a, keys_a), (best_b, second_b, keys_b)):
        for k, v in zip(keys, z, strict=True):
            if v < best.get(k, np.inf):
                second[k] = best.get(k, np.inf)
                best[k] = v
            elif v < second.get(k, np.inf):
                second[k] = v
    for n, (ka, kb) in enumerate(zip(keys_a, keys_b, strict=True)):
        # the closest rival of this pair: another b for a, or another a for b
        ra = second_a[ka] if z[n] <= best_a[ka] else best_a[ka]
        rb = second_b[kb] if z[n] <= best_b[kb] else best_b[kb]
        margin[n] = min(min(ra, rb) - z[n], MARGIN_CAP)
        mutual[n] = float(z[n] <= best_a[ka] and z[n] <= best_b[kb])
    return np.maximum(margin, -MARGIN_CAP), mutual


def features(c: Candidates, motion: MotionModel, kind: str = "movement") -> NDArray:
    """Feature matrix (columns FEATURES) for one feature set ("movement" or "fixed")."""
    if kind == "movement":
        scale = np.sqrt(c.fixed_gate_m**2 + 4.0 * motion.diffusion_m2_s * np.abs(c.dt_s))
    elif kind == "fixed":
        scale = c.fixed_gate_m
    else:
        raise ValueError(kind)
    z = c.d_m / np.maximum(scale, 1e-6)
    margin, mutual = _competition(c, z)
    return np.column_stack([z**2, margin, mutual, c.size, c.low_score, c.geotag.astype(float)])


@dataclass
class Logistic:
    mean: NDArray
    std: NDArray
    coef: NDArray  # intercept first

    def log_odds(self, x: NDArray) -> NDArray:
        return self.coef[0] + ((x - self.mean) / self.std) @ self.coef[1:]

    def prob(self, x: NDArray) -> NDArray:
        return 1.0 / (1.0 + np.exp(-self.log_odds(x)))


def fit_logistic(x: NDArray, y: NDArray, l2: float = 1e-2) -> Logistic:
    """L2-penalised logistic regression on standardised features (intercept not penalised)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    mean, std = x.mean(axis=0), x.std(axis=0)
    std = np.where(std > 1e-9, std, 1.0)
    xs = np.column_stack([np.ones(len(x)), (x - mean) / std])

    def loss(w: NDArray) -> tuple[float, NDArray]:
        t = xs @ w
        nll = np.sum(np.logaddexp(0.0, t) - y * t) / len(y)
        p = 1.0 / (1.0 + np.exp(-t))
        g = xs.T @ (p - y) / len(y)
        pen = np.r_[0.0, w[1:]]
        return nll + 0.5 * l2 * pen @ pen, g + l2 * pen

    w = minimize(loss, np.zeros(xs.shape[1]), jac=True, method="L-BFGS-B").x
    return Logistic(mean, std, w)


def auc(score: NDArray, y: NDArray) -> float:
    """Area under the ROC curve (ties counted as half)."""
    score, y = np.asarray(score, float), np.asarray(y, bool)
    pos, neg = score[y], score[~y]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.r_[pos, neg], kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.r_[pos, neg][order]
    # average ranks for ties
    _, start, counts = np.unique(allv, return_index=True, return_counts=True)
    avg = start + (counts - 1) / 2.0 + 1.0
    ranks[order] = np.repeat(avg, counts)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def match_to_labels(det_px: NDArray, lab_px: NDArray, lab_len_px: NDArray) -> NDArray:
    """Index of the label each detection matches (-1 if none): one-to-one by centre distance,
    within half the label's long side (centre matching, as in Phase 1 scoring)."""
    out = np.full(len(det_px), -1)
    if not len(det_px) or not len(lab_px):
        return out
    dist = np.linalg.norm(det_px[:, None] - lab_px[None], axis=2)
    ok = dist <= 0.5 * lab_len_px[None]
    r, c = linear_sum_assignment(np.where(ok, dist, 1e9))
    keep = ok[r, c]
    out[r[keep]] = c[keep]
    return out
