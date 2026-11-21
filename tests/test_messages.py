"""Survey result records (dashboard/schemas/survey_result.v1.json) and farmer messages."""

import json
import re
import string

import numpy as np
import pytest
from jsonschema import Draft202012Validator, FormatChecker

from herdsight.herd import messages
from herdsight.herd.alerts import Rule, run_rule
from herdsight.herd.register import HerdRegister
from herdsight.herd.sim import ErrorModel, Farm, FarmSettings, Survey, expected_table, simulate_farm

SCHEMA = json.loads(messages.SCHEMA.read_text(encoding="utf-8"))
VALIDATOR = Draft202012Validator(SCHEMA, format_checker=FormatChecker())


def valid(record):
    errors = sorted(VALIDATOR.iter_errors(record), key=str)
    assert not errors, [e.message for e in errors]


def test_schema_is_a_valid_json_schema():
    Draft202012Validator.check_schema(SCHEMA)


def test_every_message_template_is_plain():
    g = messages.glossary()
    for key, template in g["messages"].items():
        names = {f for _, f, _, _ in string.Formatter().parse(template) if f}
        text = template.format(**{n: "x" for n in names})
        assert not messages.banned_words(text), (key, messages.banned_words(text))


def test_banned_words_are_found():
    assert messages.banned_words("the model interval") == ["interval", "model"]
    with pytest.raises(ValueError):
        messages.glossary()["messages"]["zz"] = "a confidence of {camp}"
        try:
            messages.message("zz", camp="a")
        finally:
            del messages.glossary()["messages"]["zz"]


def _survey(count, lo, hi, camp="a", day=8.0, dense=False):
    return Survey(day, camp, count, count, {0.9: lo}, {0.9: hi}, dense)


def test_records_for_each_status():
    rule = Rule("recount", level=0.9)
    ok = messages.survey_record(_survey(50, 47, 53), farm_id="f1", expected=50, rule=rule)
    valid(ok)
    assert ok["status"] == "ok" and ok["alerts"] == []
    assert "50 cows counted (likely between 47 and 53)" in ok["notes"][0]

    s = _survey(40, 37.4, 43.2, dense=True)
    twice = messages.survey_record(
        s,
        farm_id="f1",
        expected=50,
        rule=rule,
        alert={"kind": "missing", "checks": 2, "other_camp": None},
        camp_names={"a": "River camp"},
    )
    valid(twice)
    a = twice["alerts"][0]
    assert twice["status"] == "missing"
    assert (a["shortfall_low"], a["shortfall_high"]) == (6, 13)  # 50 - 44 and 50 - 37
    assert (
        a["message"].startswith("Some cows may be missing in River camp")
        and "twice" in a["message"]
    )
    assert any("packed closely" in n for n in twice["notes"])

    big = messages.survey_record(
        s, farm_id="f1", expected=60, rule=rule, alert={"kind": "missing", "checks": 1}
    )
    valid(big)
    assert "straight away" in big["alerts"][0]["message"]

    moved = messages.survey_record(
        s, farm_id="f1", expected=50, rule=Rule("farm"), alert={"kind": "moved", "other_camp": "b"}
    )
    valid(moved)
    assert moved["status"] == "moved" and "may have moved" in moved["alerts"][0]["message"]

    unknown = messages.survey_record(s, farm_id="f1", expected=None, rule=rule)
    valid(unknown)
    assert unknown["status"] == "unknown"


def test_records_from_a_simulated_farm_are_valid_and_plain():
    rng = np.random.default_rng(3)
    err = ErrorModel(bias=0.05, sd=0.15, name="test")
    q = err.calibrate((0.68, 0.9, 0.95), rng)
    s = FarmSettings(200, 4, 0.05, 0.01, (3, 10), 1.0, 0.1, 0.03, 4, 6)
    farm = simulate_farm(s, err, q, rng)
    rule = Rule("farm", level=0.9)
    alerts = {(a["camp"], a["day"]): a for a in run_rule(farm, rule, err, q, rng)}
    table = expected_table(farm)
    statuses = set()
    for sv in farm.surveys:
        rec = messages.survey_record(
            sv,
            farm_id="sim1",
            expected=table[sv.camp][min(int(np.ceil(sv.day)), len(table[sv.camp]) - 1)],
            rule=rule,
            alert=alerts.get((sv.camp, sv.day)),
            count_error_model=err.name,
        )
        valid(rec)
        statuses.add(rec["status"])
        text = " ".join(rec["notes"] + [a["message"] for a in rec["alerts"]])
        assert not messages.banned_words(text)
        assert not re.search(r"\d\.\d", text)  # whole animals only
    assert "ok" in statuses and len(statuses) > 1


def test_a_record_is_marked_simulated_and_rejects_unknown_fields():
    farm = Farm(camps=["a"], register=HerdRegister(start={"a": 5}))
    assert farm.camps == ["a"]
    rec = messages.survey_record(_survey(5, 4, 6), farm_id="f", expected=5, rule=Rule("interval"))
    assert rec["simulated"] is True
    rec["extra"] = 1
    assert list(VALIDATOR.iter_errors(rec))
