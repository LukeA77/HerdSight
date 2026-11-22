"""Herd register, farm simulator and alert rules (Phase 4) on hand-made cases."""

import numpy as np
import pytest

from herdsight.herd.alerts import Rule, false_alarm_causes, loss_outcomes, run_rule, score
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


def test_false_alarm_causes():
    err = ErrorModel(bias=0.0, sd=0.0)
    rule = Rule("interval", level=0.9)
    # true 50 as recorded, counted 44: the count was wrong
    farm = _farm([(8, 44, 50)], expected=50)
    alerts = run_rule(farm, rule, err, {0.9: 2.0}, np.random.default_rng(0))
    assert [c for _, c in false_alarm_causes(farm, alerts)] == ["count_error"]
    farm.surveys[0].partial = True
    assert [c for _, c in false_alarm_causes(farm, alerts)] == ["partial_survey"]
    # 5 animals moved out on day 3 without a record: the records were out of step
    farm = _farm([(8, 45, 45)], expected=50)
    farm.unrecorded.append((3, "a", "b", 5, "move"))
    alerts = run_rule(farm, rule, err, {0.9: 2.0}, np.random.default_rng(0))
    assert [c for _, c in false_alarm_causes(farm, alerts)] == ["unrecorded_move"]
    farm.losses = [Loss(5, "a", 1, "death")]  # with a real loss the alert is not false
    assert false_alarm_causes(farm, alerts) == []


def test_loss_outcomes_agree_with_score():
    rng = np.random.default_rng(4)
    err = ErrorModel(bias=0.0, sd=0.1)
    q = err.calibrate(LEVELS, rng)
    s = FarmSettings(300, 4, 0.1, 0.01, (3, 12), 1.0, 0.1, 0.03, 4, 12)
    farm = simulate_farm(s, err, q, rng)
    alerts = run_rule(farm, Rule("recount", level=0.9), err, q, rng)
    rows = loss_outcomes(farm, alerts, 3)
    for k in (1, 5):
        r = score(farm, alerts, k, 3)
        mine = [x for x in rows if x["n"] >= k]
        assert len(mine) == r["losses"] and sum(x["caught"] for x in mine) == r["caught"]
        assert sorted(x["surveys_until_caught"] for x in mine if x["caught"]) == sorted(
            r["surveys_until_caught"]
        )


def test_simulator_labels_add_up():
    rng = np.random.default_rng(5)
    err = ErrorModel(bias=0.0, sd=0.1)
    s = FarmSettings(400, 5, 0.06, 0.01, (1, 20), 2.0, 0.2, 0.05, 4, 12)
    farm = simulate_farm(s, err, err.calibrate(LEVELS, rng), rng)
    kinds = {u[4] for u in farm.unrecorded}
    assert kinds == {"move", "stray"} and farm.register_errors
    # records minus truth over the whole farm = losses + register errors (moves and strays net out)
    last = len(farm.true_log) - 1
    table = expected_table(farm)
    gap = sum(table[c][last] for c in farm.camps) - sum(farm.true_log[last][1].values())
    lost = sum(x.n for x in farm.losses)
    assert gap == lost + sum(n for d, _, n in farm.register_errors if d <= last)


def test_stricter_shortcut_asks_for_a_recount():
    # counted 30 against 50 recorded; the 90% range (to 38) is 12 below, the 99% range (to 45) 5
    farm = _farm([(8, 30, 50)], expected=50)
    s = farm.surveys[0]
    s.lo, s.hi = {0.9: 22.0, 0.99: 15.0}, {0.9: 38.0, 0.99: 45.0}
    err = ErrorModel(bias=0.0, sd=0.0)
    q = {0.9: 0.01, 0.99: 0.02}  # the recount sees the true 50: no shortfall
    rng = np.random.default_rng(0)
    at_once = run_rule(farm, Rule("recount", level=0.9), err, q, rng)
    assert [a["checks"] for a in at_once] == [1]  # 12 >= 10: alerted without a recount
    strict = run_rule(farm, Rule("recount", level=0.9, big_level=0.99), err, q, rng)
    assert strict == []  # 5 < 10: recounted, and the recount found them all


def test_history_filter_pools_surveys_and_resets():
    from herdsight.herd.alerts import ShortfallFilter

    f = ShortfallFilter(drift=0.2)
    s = Survey(1, "a", 50, 45, {0.68: 42.0, 0.9: 40.0}, {0.68: 48.0, 0.9: 50.0}, False)
    assert f.update(s, 50)[0] < 1  # one survey of a 5-animal shortfall barely moves it
    for d in range(2, 12):
        s.day = d
        m, v = f.update(s, 50)
    assert 3 < m < 5 and v < 9  # repeated surveys pool towards the shortfall, tighter than one
    assert f.short(0.68)
    f.reset(12)
    assert (f.mean, f.var) == (0.0, 0.25) and not f.short(0.68)


def test_history_rule_runs_and_needs_agreement():
    farm = _farm([(8, 44, 50)], expected=50)
    for s in farm.surveys:
        s.lo[0.68], s.hi[0.68] = s.count - 1.0, s.count + 1.0
    err = ErrorModel(bias=0.0, sd=0.0)
    q = {0.68: 0.0, 0.9: 0.0}  # recounts see the true 50: no shortfall
    alerts = run_rule(farm, Rule("history", level=0.68), err, q, np.random.default_rng(0))
    assert alerts == []  # one low survey, then a recount that finds them all: no alert
    farm = _farm([(8, 44, 44)], expected=50)  # 6 really gone; the recount agrees
    farm.true_log = [(d, {"a": 44}) for d in range(10)]
    farm.surveys[0].lo[0.68], farm.surveys[0].hi[0.68] = 43.0, 45.0
    alerts = run_rule(farm, Rule("history", level=0.68), err, q, np.random.default_rng(0))
    assert [a["checks"] for a in alerts] == [2]
