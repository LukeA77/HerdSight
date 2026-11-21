"""Missing-animal alert rules and their S4 scores (Phase 4 plan, steps 4-5; D-13).

Rules (each a function of the surveys in time order and the register):
  naive      alert when the count is below the expected count;
  interval   alert when the whole interval at `level` is below the expected count (D-13);
  recount    a first interval shortfall triggers a recount (herd mode the next day); the alert
             needs the recount to agree, unless the shortfall (expected - upper end) is at least
             `big` animals, which alerts at once;
  farm       as recount, but a camp shortfall matched by a surplus in another camp surveyed within
             `farm_window` days is reported as "may have moved", not as a loss.
After an alert, or a "may have moved" notice, the farmer checks and the register is corrected to
the true count for that camp (a CORRECTION event), as the dashboard's "Resolved" action would.

Scoring (S4): an alert is a true alarm if, when it fires, the camp has at least one animal lost
(death or theft) since its register was last corrected; otherwise it is a false alarm (unrecorded
moves, strays, register errors and count errors cause these). A loss is caught if an alert on its
camp fires within `within_surveys` surveys of that camp after the loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from herdsight.herd.sim import ErrorModel, Farm, Survey, expected_table, survey


@dataclass(frozen=True)
class Rule:
    name: str  # naive | interval | recount | farm
    level: float = 0.9
    big: float = 10.0
    farm_window: float = 7.0


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
    for s in sorted(farm.surveys, key=lambda x: (x.day, x.camp)):
        recent[s.camp] = s
        exp = expected(s.camp, s.day)
        if not _short(s, exp, rule):
            continue
        kind, checks, other = "missing", 1, None
        if rule.name in ("recount", "farm") and exp - s.hi[rule.level] < rule.big:
            day = s.day + 1
            t = true_by_day[min(int(day), len(true_by_day) - 1)][1][s.camp]
            again = survey(day, s.camp, t, err, q, {s.camp: s.z}, rng)
            if not _short(again, expected(s.camp, day), rule):
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
