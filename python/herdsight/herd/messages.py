"""Survey result records and farmer-facing messages (Phase 4 plan, steps 2 and 5; D-17).

`survey_record` turns one survey of one camp, and any alert it raised, into a record matching
dashboard/schemas/survey_result.v1.json, the only thing the dashboard reads. Every sentence in a
record comes from the templates in dashboard/glossary.yaml, so technical terms never reach the
farmer and the wording can be translated without touching code.
"""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any

import yaml

from herdsight.herd.alerts import Rule
from herdsight.herd.sim import Survey
from herdsight.utils.config import REPO_ROOT

GLOSSARY = REPO_ROOT / "dashboard" / "glossary.yaml"
SCHEMA = REPO_ROOT / "dashboard" / "schemas" / "survey_result.v1.json"


@lru_cache(maxsize=1)
def glossary() -> dict[str, Any]:
    return yaml.safe_load(GLOSSARY.read_text(encoding="utf-8"))


def banned_words(text: str) -> list[str]:
    """Technical words from the glossary's banned list that appear in `text`."""
    return [
        w for w in glossary()["banned_in_messages"] if re.search(rf"\b{w}\w*", text, re.IGNORECASE)
    ]


def message(key: str, **fields: Any) -> str:
    g = glossary()
    fields.setdefault("animals", g["animals"]["cattle"])
    text = g["messages"][key].format(**fields)
    bad = banned_words(text)
    if bad:
        raise ValueError(f"technical words {bad} in farmer message {key!r}")
    return text


def _range(s: Survey, level: float) -> tuple[int, int]:
    return max(0, math.floor(s.lo[level])), max(0, math.ceil(s.hi[level]))


def survey_record(
    s: Survey,
    *,
    farm_id: str,
    expected: float | None,
    rule: Rule,
    alert: dict | None = None,
    camp_names: dict[str, str] | None = None,
    start: datetime = datetime(2026, 1, 1, tzinfo=UTC),
    count_error_model: str = "",
    pipeline_version: str = "sim",
    simulated: bool = True,
    mode: str = "survey",
) -> dict[str, Any]:
    """One schema record for survey `s`; `alert` is the run_rule record it raised, if any."""
    names = camp_names or {}
    camp = names.get(s.camp, s.camp)
    level = rule.level
    low, high = _range(s, level)
    exp = None if expected is None else int(round(expected))
    notes = []
    if s.dense:
        notes.append(message("dense_note"))
    alerts = []
    if exp is None:
        status = "unknown"
    elif alert is None:
        status = "ok"
    else:
        status = alert["kind"]
        short_low, short_high = max(0, exp - high), max(0, exp - low)
        fields = dict(camp=camp, expected=exp, short_low=short_low, short_high=short_high)
        if alert["kind"] == "moved":
            other = alert.get("other_camp")
            text = message("moved", other_camp=names.get(other, other), **fields)
        elif alert.get("checks", 1) >= 2:
            text = message("missing_checked_twice", **fields)
        else:
            text = message("missing_large", **fields)
        alerts.append(
            {
                "alert_id": f"{farm_id}-{s.camp}-{s.day:g}",
                "kind": alert["kind"],
                "message": text,
                "shortfall_low": short_low,
                "shortfall_high": short_high,
                "checks": int(alert.get("checks", 1)),
                "other_camp_id": alert.get("other_camp"),
                "rule": {"name": rule.name, "level": level, "large_loss": rule.big},
            }
        )
    if status == "ok":
        notes.insert(0, message("ok", camp=camp, count=s.count, low=low, high=high, expected=exp))
    return {
        "schema_version": "1",
        "survey_id": f"{farm_id}-{s.camp}-{s.day:g}-{mode}",
        "farm_id": farm_id,
        "camp_id": s.camp,
        "camp_name": camp,
        "surveyed_at": (start + timedelta(days=float(s.day))).isoformat(),
        "mode": mode,
        "simulated": simulated,
        "count": int(s.count),
        "intervals": [
            {"level": lv, "low": _range(s, lv)[0], "high": _range(s, lv)[1]} for lv in sorted(s.lo)
        ],
        "dense_herd": bool(s.dense),
        "coverage": None,
        "expected_count": exp,
        "status": status,
        "alerts": alerts,
        "notes": notes,
        "evidence": [],
        "pipeline": {"version": pipeline_version, "count_error_model": count_error_model},
    }
