"""Herd register, farm simulator and alert rules (Phase 4) on hand-made cases."""

import numpy as np
import pytest

from herdsight.herd.alerts import Rule, run_rule, score
from herdsight.herd.register import Event, HerdRegister, Kind
from herdsight.herd.sim import (
    ErrorModel,
    Farm,
    FarmSettings,
    Loss,
    Survey,
    expected_table,
    simulate_farm,
)

LEVELS = (0.68, 0.9, 0.95)


def test_register_events_and_moves_keep_the_farm_total():
    r = HerdRegister(start={"a": 10, "b": 5})
    r.record(Event(1, Kind.BIRTH, "a", 2))
    r.record(Event(2, Kind.MOVE, "a", 4, to_camp="b"))
    r.record(Event(3, Kind.SALE, "b", 1))
    assert r.expected("a", 0) == 10
    assert r.expected("a", 1) == 12 and r.expected("a", 2) == 8 and r.expected("b", 2) == 9
    assert r.farm_total(2) == 17 and r.farm_total(3) == 16  # the move changed nothing
    with pytest.raises(ValueError):
        Event(1, Kind.MOVE, "a", 1)  # a move needs a destination
    with pytest.raises(KeyError):
        r.record(Event(1, Kind.BIRTH, "zzz", 1))


def test_expected_table_matches_the_register():
    rng = np.random.default_rng(0)
    err = ErrorModel(bias=0.0, sd=0.05)
    s = FarmSettings(120, 3, 0.05, 0.005, (1, 5), 1.0, 0.2, 0.05, 4, 3)
    farm = simulate_farm(s, err, err.calibrate(LEVELS, rng), rng)
    table = expected_table(farm)
    for day in (0, 17, 45, 89):
        for c in farm.camps:
            assert table[c][day] == farm.register.expected(c, day)


def test_calibrated_intervals_cover_at_their_level():
    rng = np.random.default_rng(1)
    err = ErrorModel(bias=0.05, sd=0.15, partial_prob=0.05)
    q = err.calibrate(LEVELS, rng)
    s = FarmSettings(300, 4, 0.0, 0.0, (1, 1), 0.0, 0.0, 0.0, 4, 12)
    farm = simulate_farm(s, err, q, rng)
    for lv in LEVELS:
        inside = np.mean([sv.lo[lv] <= sv.true <= sv.hi[lv] for sv in farm.surveys])
        assert abs(inside - lv) < 0.06, (lv, inside)


def _farm(counts, expected, losses=()):
    """A farm with one camp, hand-made surveys and a flat register."""
    reg = HerdRegister(start={"a": expected})
    days = int(max(c[0] for c in counts)) + 2
    farm = Farm(camps=["a"], register=reg)
    farm.true_log = [(d, {"a": expected}) for d in range(days)]
    for day, count, true in counts:
        half = 2.0
        farm.surveys.append(
            Survey(day, "a", true, count, {0.9: count - half}, {0.9: count + half}, False)
        )
    farm.losses = list(losses)
    return farm


def test_naive_and_interval_rules_and_scoring():
    # true count stays 50 (no loss); the count dips by 2 once from error alone
    farm = _farm([(1, 50, 50), (8, 48, 50), (15, 50, 50)], expected=50)
    err = ErrorModel(bias=0.0, sd=0.0)
    rng = np.random.default_rng(0)
    naive = run_rule(farm, Rule("naive"), err, {0.9: 2.0}, rng)
    interval = run_rule(farm, Rule("interval", level=0.9), err, {0.9: 2.0}, rng)
    assert [a["day"] for a in naive] == [8]  # a false alarm
    assert interval == []  # 48 + 2 = 50 is not below the expected 50: no alert
    r = score(farm, naive, k=1, within_surveys=2)
    assert r["false_alarms"] == 1 and r["true_alarms"] == 0


def test_a_real_loss_is_caught_and_scored():
    # 6 animals stolen on day 5: true count 44 from then on
    farm = _farm(
        [(1, 50, 50), (8, 44, 44), (15, 44, 44)], expected=50, losses=[Loss(5, "a", 6, "theft")]
    )
    err = ErrorModel(bias=0.0, sd=0.0)
    alerts = run_rule(farm, Rule("interval", level=0.9), err, {0.9: 2.0}, np.random.default_rng(0))
    assert [a["day"] for a in alerts] == [8]
    r = score(farm, alerts, k=5, within_surveys=2)
    assert (r["true_alarms"], r["false_alarms"], r["losses"], r["caught"]) == (1, 0, 1, 1)
    assert r["surveys_until_caught"] == [1]
