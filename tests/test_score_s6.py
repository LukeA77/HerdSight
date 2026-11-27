"""S6 scoring applies the pass rule fixed before the sessions (docs/plans/s6-usability-test.md)."""

from __future__ import annotations

import importlib.util

from herdsight.utils.config import REPO_ROOT

spec = importlib.util.spec_from_file_location("score_s6", REPO_ROOT / "scripts" / "score_s6.py")
s6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s6)

BASE = {
    "tester": "Jane Doe",
    "date": "2026-11-28",
    "task1_camp": "Bush camp",
    "task1_seconds": "6",
    "task2_said": "",
    "task3_without_help": "yes",
    "task3_seconds": "25",
    "task4_camps": "",
    "task5_worked_offline": "yes",
    "confusing": "",
    "trust": "",
}


def test_pass_needs_bush_camp_quickly_and_the_alert_handled_alone() -> None:
    assert s6.score_row(BASE)["s6_pass"]
    assert not s6.score_row(BASE | {"task1_camp": "Top camp"})["s6_pass"]
    assert not s6.score_row(BASE | {"task1_seconds": "11"})["s6_pass"]
    assert not s6.score_row(BASE | {"task3_without_help": "no"})["s6_pass"]
    assert not s6.score_row(BASE | {"task3_seconds": ""})["s6_pass"]


def test_only_initials_are_kept() -> None:
    assert s6.score_row(BASE)["tester"] == "JD"
