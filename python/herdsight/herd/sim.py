"""Simulated farms and surveys for testing alert rules (Phase 4 plan, step 3).

A farm has camps with a true number of animals that changes day by day. Some changes are recorded
in the herd register (births, sales, recorded moves), some are not:
  * real losses: deaths and thefts, which alerts should catch;
  * unrecorded moves and straying, where animals are in another camp, not lost;
  * register errors, where a recorded number is wrong.
Each camp is surveyed on a schedule. A survey's count has a relative error drawn from an error
model fitted to measured count errors; successive errors in the same camp can be correlated; part
of a camp may go unsurveyed (undercount); dense herds have larger errors. Each count carries a
calibrated interval by split conformal prediction on the relative residual
|true - predicted| / (predicted + 1), fitted on a calibration set drawn from the same error model.
(Flight-total errors grow in proportion to the herd, so the relative residual is the matching
score here; per-image counts in the detector pipeline use a square-root score instead.)

All randomness comes from one numpy Generator, so a seed reproduces a farm exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from herdsight.herd.register import Event, HerdRegister, Kind


@dataclass(frozen=True)
class ErrorModel:
    """Relative count error per survey: bias + sd * z, z correlated over a camp's surveys."""

    bias: float
    sd: float
    correlation: float = 0.0  # AR(1) correlation between successive surveys of a camp
    dense_sd_factor: float = 1.5  # sd multiplier when the camp holds >= dense_at animals
    dense_at: int = 20
    partial_prob: float = 0.05  # chance a survey misses part of the camp
    partial_coverage: tuple[float, float] = (0.6, 0.95)
    name: str = ""

    def calibrate(self, levels: tuple[float, ...], rng: np.random.Generator) -> dict[float, float]:
        """Conformal quantile of |true - pred| / (pred + 1) for each level."""
        true = rng.integers(5, 300, 20000).astype(float)
        sd = np.where(true >= self.dense_at, self.sd * self.dense_sd_factor, self.sd)
        pred = np.maximum(np.round(true * (1 + self.bias + sd * rng.standard_normal(true.size))), 0)
        partial = rng.random(true.size) < self.partial_prob
        pred[partial] = np.round(pred[partial] * rng.uniform(*self.partial_coverage, partial.sum()))
        score = np.abs(true - pred) / (pred + 1)
        n = score.size
        return {lv: float(np.quantile(score, min(1.0, np.ceil((n + 1) * lv) / n))) for lv in levels}


@dataclass(frozen=True)
class FarmSettings:
    herd_size: int
    camps: int
    annual_mortality: float
    theft_share_per_year: float
    theft_size: tuple[int, int]
    straying_per_camp_year: float
    unrecorded_move_share: float
    register_error_share: float
    surveys_per_month: int
    months: int


@dataclass
class Survey:
    day: float
    camp: str
    true: int
    count: int
    lo: dict[float, float]
    hi: dict[float, float]
    dense: bool
    z: float = 0.0  # the survey's standardised error, for correlated recounts
    partial: bool = False  # part of the camp was not seen


@dataclass
class Loss:
    day: float
    camp: str
    n: int
    kind: str  # death or theft


@dataclass
class Farm:
    camps: list[str]
    register: HerdRegister
    true_log: list[tuple[float, dict[str, int]]] = field(default_factory=list)
    losses: list[Loss] = field(default_factory=list)
    surveys: list[Survey] = field(default_factory=list)
    # unrecorded changes: (day, from camp, to camp, animals, "move" or "stray")
    unrecorded: list[tuple[float, str, str, int, str]] = field(default_factory=list)
    register_errors: list[tuple[float, str, int]] = field(default_factory=list)  # (day, camp, +-)


def simulate_farm(
    s: FarmSettings,
    err: ErrorModel,
    q: dict[float, float],
    rng: np.random.Generator,
    obs_rng: np.random.Generator | None = None,
) -> Farm:
    """One farm's history: daily true counts, register entries, losses and surveys.

    `rng` drives what happens on the farm; `obs_rng` (default: the same generator) drives survey
    errors, so the same farm history can be surveyed under different error models."""
    obs = obs_rng if obs_rng is not None else rng
    camps = [f"camp{i + 1}" for i in range(s.camps)]
    split = rng.multinomial(s.herd_size, rng.dirichlet(np.full(s.camps, 2.0)))
    true = {c: int(n) for c, n in zip(camps, split, strict=True)}
    reg = HerdRegister(start=dict(true))
    farm = Farm(camps=camps, register=reg)
    days = int(s.months * 30)
    every = 30.0 / s.surveys_per_month
    next_survey = {c: rng.uniform(0, every) for c in camps}
    z_prev = {c: obs.standard_normal() for c in camps}
    strays: list[tuple[int, str, str, int]] = []  # (return day, from, to, n)
    rate_day = 1.0 / 365.0

    def bad(n: int) -> int:  # a register error: wrong number entered
        return max(0, n + int(rng.choice((-2, -1, 1, 2))))

    for day in range(days):
        # deaths (real losses, not in the register)
        for c in camps:
            d = rng.binomial(true[c], s.annual_mortality * rate_day)
            if d:
                true[c] -= d
                farm.losses.append(Loss(day, c, int(d), "death"))
        # theft (real loss of several animals at once)
        mean_size = (s.theft_size[0] + s.theft_size[1]) / 2
        p_theft = s.theft_share_per_year * sum(true.values()) / max(mean_size, 1) * rate_day
        if rng.random() < p_theft:
            c = camps[int(rng.integers(s.camps))]
            k = int(min(true[c], rng.integers(s.theft_size[0], s.theft_size[1] + 1)))
            if k:
                true[c] -= k
                farm.losses.append(Loss(day, c, k, "theft"))
        # births and sales, recorded (with occasional register errors)
        for c in camps:
            b = rng.binomial(true[c], 0.8 * rate_day)  # about one calf per adult per year
            if b:
                true[c] += b
                rec = bad(b) if rng.random() < s.register_error_share else b
                reg.record(Event(day, Kind.BIRTH, c, rec))
                if rec != b:
                    farm.register_errors.append((day, c, rec - b))
            sold = rng.binomial(true[c], 0.6 * rate_day)
            if sold:
                true[c] -= sold
                rec = bad(sold) if rng.random() < s.register_error_share else sold
                reg.record(Event(day, Kind.SALE, c, rec))
                if rec != sold:
                    farm.register_errors.append((day, c, sold - rec))
        # moves between camps (sometimes unrecorded)
        if s.camps > 1 and rng.random() < 2.0 * rate_day * s.camps:
            a, b2 = rng.choice(s.camps, 2, replace=False)
            ca, cb = camps[int(a)], camps[int(b2)]
            k = int(true[ca] * rng.uniform(0.1, 0.5))
            if k:
                true[ca] -= k
                true[cb] += k
                if rng.random() < s.unrecorded_move_share:
                    farm.unrecorded.append((day, ca, cb, k, "move"))
                else:
                    reg.record(Event(day, Kind.MOVE, ca, k, to_camp=cb))
        # straying through fences, returning after 3-30 days
        if s.camps > 1:
            for c in camps:
                if true[c] and rng.random() < s.straying_per_camp_year * rate_day:
                    to = camps[int(rng.integers(s.camps))]
                    if to != c:
                        k = int(min(true[c], rng.integers(1, 4)))
                        true[c] -= k
                        true[to] += k
                        strays.append((day + int(rng.integers(3, 31)), c, to, k))
                        farm.unrecorded.append((day, c, to, k, "stray"))
        for r in [x for x in strays if x[0] == day]:
            _, c, to, k = r
            k = min(k, true[to])
            true[to] -= k
            true[c] += k
            strays.remove(r)
        farm.true_log.append((day, dict(true)))
        # surveys
        for c in camps:
            if day < next_survey[c]:
                continue
            next_survey[c] += every
            farm.surveys.append(survey(day, c, true[c], err, q, z_prev, obs))
    return farm


def survey(
    day: float,
    camp: str,
    true: int,
    err: ErrorModel,
    q: dict[float, float],
    z_prev: dict[str, float],
    rng: np.random.Generator,
) -> Survey:
    """One survey of one camp: a count with an error and calibrated intervals."""
    rho = err.correlation
    z = rho * z_prev[camp] + np.sqrt(1 - rho * rho) * rng.standard_normal()
    z_prev[camp] = z
    dense = true >= err.dense_at
    sd = err.sd * (err.dense_sd_factor if dense else 1.0)
    count = max(0, int(round(true * (1 + err.bias + sd * z))))
    partial = bool(rng.random() < err.partial_prob)
    if partial:
        count = int(round(count * rng.uniform(*err.partial_coverage)))
    half = {lv: qq * (count + 1) for lv, qq in q.items()}
    return Survey(
        day=day,
        camp=camp,
        true=true,
        count=count,
        lo={lv: max(0.0, count - h) for lv, h in half.items()},
        hi={lv: count + h for lv, h in half.items()},
        dense=bool(dense),
        z=float(z),
        partial=partial,
    )


def draw_settings(ranges: dict, rng: np.random.Generator) -> FarmSettings:
    """One farm's settings, drawn uniformly from the ranges in configs/sim_farms.yaml."""

    def u(key: str) -> float:
        lo, hi = ranges[key]
        return float(rng.uniform(lo, hi))

    lo_c, hi_c = ranges["camps_per_farm_range"]
    lo_s, hi_s = ranges["surveys_per_month"]
    return FarmSettings(
        herd_size=int(round(np.exp(rng.uniform(*np.log(ranges["herd_size_range"]))))),
        camps=int(rng.integers(lo_c, hi_c + 1)),
        annual_mortality=u("annual_mortality_range"),
        theft_share_per_year=u("theft_share_per_year"),
        theft_size=tuple(ranges["theft_event_size_range"]),
        straying_per_camp_year=u("straying_events_per_camp_year"),
        unrecorded_move_share=u("unrecorded_move_share"),
        register_error_share=u("register_error_share"),
        surveys_per_month=int(rng.integers(lo_s, hi_s + 1)),
        months=int(ranges["months_per_farm"]),
    )


def expected_table(farm: Farm) -> dict[str, np.ndarray]:
    """Register expected count per camp for every simulated day (fast lookup for alert rules)."""
    days = len(farm.true_log)
    out = {c: np.full(days, float(farm.register.start[c])) for c in farm.camps}
    for e in farm.register.events:
        d = int(np.ceil(e.day))
        if d >= days:
            continue
        sign = {Kind.BIRTH: 1, Kind.PURCHASE: 1, Kind.SALE: -1, Kind.DEATH: -1}.get(e.kind)
        if sign is not None:
            out[e.camp][d:] += sign * e.n
        elif e.kind is Kind.CORRECTION:
            out[e.camp][d:] += e.n
        elif e.kind is Kind.MOVE:
            out[e.camp][d:] -= e.n
            out[e.to_camp][d:] += e.n
    return out


def measured_flight_error(source: str = "results/baselines/dfine_s3.json") -> tuple[float, float]:
    """(bias, sd) of the relative error of flight totals after duplicate removal on the
    owner-verified flights in `source` (registration method). The default is the Phase 1
    detector on the first ten verified flights; results/phase3/dfine_ref_s3_v20.json is the final
    detector on all twenty."""
    import json

    from herdsight.utils.config import REPO_ROOT

    s3 = json.loads((REPO_ROOT / source).read_text())
    rel = np.array(
        [
            f["relative_error"]["registration"]
            for f in s3["flights"].values()
            if f["truth_kind"] == "owner-verified"
        ]
    )
    return float(rel.mean()), float(rel.std(ddof=1))
