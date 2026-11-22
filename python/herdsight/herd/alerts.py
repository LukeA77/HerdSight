"""Missing-animal alert rules and their S4 scores (Phase 4 plan, steps 4-5; D-13).

Rules (each a function of the surveys in time order and the register):
  naive      alert when the count is below the expected count;
  interval   alert when the whole interval at `level` is below the expected count (D-13);
  recount    a first interval shortfall triggers a recount (herd mode the next day); the alert
             needs the recount to agree, unless the shortfall (expected - upper end) is at least
             `big` animals, which alerts at once (the upper end of the interval at `big_level`,
             if set, for this shortcut);
  farm       as recount, but a camp shortfall matched by a surplus in another camp surveyed within
             `farm_window` days is reported as "may have moved", not as a loss;
  history    as recount, but a shortfall is judged from every survey of the camp since its records
             were last confirmed: a one-dimensional Kalman filter of the camp's shortfall
             (expected minus true count). Each survey's error standard deviation is the half-width
             of its 68% interval; before each survey is used, the shortfall's variance grows by
             `drift` times that survey's error variance (losses and unrecorded changes), so the
             filter weighs about the last three surveys whatever the herd size; after a correction
             the shortfall is known to within half an animal. A shortfall is flagged when the
             filter's lower bound at `level` is above half an animal (H18-H19).
After an alert, or a "may have moved" notice, the farmer checks and the register is corrected to
the true count for that camp (a CORRECTION event), as the dashboard's "Resolved" action would.

Scoring (S4): an alert is a true alarm if, when it fires, the camp has at least one animal lost
(death or theft) since its register was last corrected; otherwise it is a false alarm (unrecorded
moves, strays, register errors and count errors cause these). A loss is caught if an alert on its
camp fires within `within_surveys` surveys of that camp after the loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from herdsight.herd.sim import ErrorModel, Farm, Survey, expected_table, survey


@dataclass(frozen=True)
class Rule:
    name: str  # naive | interval | recount | farm | history
    level: float = 0.9
    big: float = 10.0
    farm_window: float = 7.0
    big_level: float | None = None  # interval level for the large-loss shortcut (None: `level`)
    drift: float = 0.2  # history rule: variance added per survey, share of its error variance


def _z(level: float) -> float:
    return NormalDist().inv_cdf((1 + level) / 2)


def _survey_sd(s: Survey) -> float:
    """Error standard deviation of a survey count, from its 68% interval (or another level)."""
    lv = 0.68 if 0.68 in s.hi else min(s.hi)
    return max(0.5, (s.hi[lv] - s.count) / _z(lv))


class ShortfallFilter:
    """Kalman filter of one camp's shortfall (expected minus true count) between corrections."""

    def __init__(self, drift: float, day: float = 0.0) -> None:
        self.drift = drift
        self.reset(day)

    def reset(self, day: float) -> None:
        self.mean, self.var, self.day = 0.0, 0.25, day  # known to within half an animal

    def update(self, s: Survey, expected: float) -> tuple[float, float]:
        r = _survey_sd(s) ** 2
        var = self.var + self.drift * r
        gain = var / (var + r)
        self.mean += gain * ((expected - s.count) - self.mean)
        self.var, self.day = (1 - gain) * var, s.day
        return self.mean, self.var

    def short(self, level: float) -> bool:
        return self.mean - _z(level) * self.var**0.5 > 0.5


def _short(s: Survey, expected: int, rule: Rule) -> bool:
    if rule.name == "naive":
        return s.count < expected
    return s.hi[rule.level] < expected


def run_rule(
    farm: Farm, rule: Rule, err: ErrorModel, q: dict[float, float], rng: np.random.Generator
) -> list[dict]:
    """Apply a rule to a farm's surveys; returns alert records in time order.

    The farm's own register is not changed: corrections after each alert are kept per camp here,
    so several rules can be run on the same farm."""
    base = expected_table(farm)
    fixes: dict[str, list[tuple[float, float]]] = {c: [] for c in farm.camps}

    def expected(camp: str, day: float) -> float:
        d = min(int(np.ceil(day)), len(base[camp]) - 1)
        return base[camp][d] + sum(delta for when, delta in fixes[camp] if when <= day)

    true_by_day = farm.true_log
    alerts: list[dict] = []
    recent: dict[str, Survey] = {}
    history = {c: ShortfallFilter(rule.drift) for c in farm.camps}
    for s in sorted(farm.surveys, key=lambda x: (x.day, x.camp)):
        recent[s.camp] = s
        exp = expected(s.camp, s.day)
        if rule.name == "history":
            history[s.camp].update(s, exp)
            if not history[s.camp].short(rule.level):
                continue
        elif not _short(s, exp, rule):
            continue
        kind, checks, other = "missing", 1, None
        shortcut_level = rule.big_level or rule.level
        if rule.name in ("recount", "farm", "history") and exp - s.hi[shortcut_level] < rule.big:
            day = s.day + 1
            t = true_by_day[min(int(day), len(true_by_day) - 1)][1][s.camp]
            again = survey(day, s.camp, t, err, q, {s.camp: s.z}, rng)
            if rule.name == "history":
                history[s.camp].update(again, expected(s.camp, day))
                if not history[s.camp].short(rule.level):
                    continue
            elif not _short(again, expected(s.camp, day), rule):
                continue
            checks = 2
        if rule.name == "farm":
            gap = exp - s.hi[rule.level]
            for c, o in recent.items():
                if c == s.camp or abs(o.day - s.day) > rule.farm_window:
                    continue
                surplus = o.lo[rule.level] - expected(c, o.day)
                if surplus > 0 and surplus >= 0.5 * gap:
                    kind, other = "moved", c
                    break
        alerts.append(
            {
                "day": s.day,
                "camp": s.camp,
                "kind": kind,
                "expected": exp,
                "count": s.count,
                "checks": checks,
                "other_camp": other,
                "survey": s,
            }
        )
        # the farmer checks; the register is set to the camp's true count
        fixes[s.camp].append((s.day, s.true - expected(s.camp, s.day)))
        history[s.camp].reset(s.day)
    return alerts


def score(farm: Farm, alerts: list[dict], k: int, within_surveys: int) -> dict:
    """False alarms, real alarms and losses caught for one farm."""
    corrections = {c: [] for c in farm.camps}
    for a in alerts:
        corrections[a["camp"]].append(a["day"])
    false_alarms = true_alarms = 0
    for a in alerts:
        prev = max([d for d in corrections[a["camp"]] if d < a["day"]], default=-1)
        lost = sum(x.n for x in farm.losses if x.camp == a["camp"] and prev < x.day <= a["day"])
        if a["kind"] == "missing" and lost > 0:
            true_alarms += 1
        elif a["kind"] == "missing":
            false_alarms += 1
    by_camp = {c: sorted(s.day for s in farm.surveys if s.camp == c) for c in farm.camps}
    caught = total = 0
    delays = []
    for x in farm.losses:
        if x.n < k:
            continue
        later = [d for d in by_camp[x.camp] if d >= x.day][:within_surveys]
        if len(later) < within_surveys:
            continue  # the history ends before the window closes: not scored
        total += 1
        hit = [a["day"] for a in alerts if a["camp"] == x.camp and x.day <= a["day"] <= later[-1]]
        if hit:
            caught += 1
            delays.append(sum(1 for d in later if d <= min(hit)))
    return {
        "surveys": len(farm.surveys),
        "false_alarms": false_alarms,
        "true_alarms": true_alarms,
        "moved_notices": sum(a["kind"] == "moved" for a in alerts),
        "losses": total,
        "caught": caught,
        "surveys_until_caught": delays,
    }


def loss_outcomes(farm: Farm, alerts: list[dict], within_surveys: int) -> list[dict]:
    """Each scored loss with whether it was caught, for breakdowns; same rules as `score`."""
    by_camp = {c: sorted(s.day for s in farm.surveys if s.camp == c) for c in farm.camps}
    true_at = farm.true_log
    out = []
    for x in farm.losses:
        later = [d for d in by_camp[x.camp] if d >= x.day][:within_surveys]
        if len(later) < within_surveys:
            continue
        hit = [a["day"] for a in alerts if a["camp"] == x.camp and x.day <= a["day"] <= later[-1]]
        before = true_at[int(x.day) - 1][1][x.camp] if x.day >= 1 else farm.register.start[x.camp]
        out.append(
            {
                "kind": x.kind,
                "n": x.n,
                "camp_animals": int(before),
                "caught": bool(hit),
                "surveys_until_caught": sum(1 for d in later if d <= min(hit)) if hit else None,
            }
        )
    return out


FALSE_ALARM_CAUSES = (
    "unrecorded_move",
    "stray",
    "register_error",
    "partial_survey",
    "dense_herd",
    "count_error",
    "other_records",
)


def false_alarm_causes(farm: Farm, alerts: list[dict]) -> list[tuple[dict, str]]:
    """Each false "missing" alert with its most likely cause, for breakdowns.

    If the camp's true number was below the expected count when the alert fired, the records were
    out of step: the cause is an unrecorded move, a stray or a register error in that camp since
    its last correction (the largest by animals), or strays that were in the camp and have gone
    home; "other_records" if none is found. Otherwise the count was wrong: a partial survey,
    a dense herd, or ordinary count error, in that order."""
    last: dict[str, float] = {c: -1.0 for c in farm.camps}
    out = []
    for a in alerts:
        camp, day, prev = a["camp"], a["day"], last[a["camp"]]
        last[camp] = day
        lost = any(x.camp == camp and prev < x.day <= day for x in farm.losses)
        if a["kind"] != "missing" or lost:
            continue
        s = a["survey"]
        if a["expected"] - s.true >= 1:
            weight = dict.fromkeys(FALSE_ALARM_CAUSES[:3], 0)
            for d, frm, to, n, kind in farm.unrecorded:
                if frm == camp and prev < d <= day:
                    weight["unrecorded_move" if kind == "move" else "stray"] += n
                elif kind == "stray" and to == camp and d <= day:
                    weight["stray"] += n  # strays counted here earlier have gone home
            for d, c, n in farm.register_errors:
                if c == camp and prev < d <= day and n > 0:
                    weight["register_error"] += n
            cause = max(weight, key=weight.get) if any(weight.values()) else "other_records"
        elif s.partial:
            cause = "partial_survey"
        elif s.dense:
            cause = "dense_herd"
        else:
            cause = "count_error"
        out.append((a, cause))
    return out
