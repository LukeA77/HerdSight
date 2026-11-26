"""SQLite store for survey result records, farm layouts and the farmer's actions on alerts.

Every record is checked against dashboard/schemas/survey_result.v1.json before it is stored, so the
dashboard can trust what it reads. A farm layout gives each camp a name and an outline (metres on a
local east-north grid) for the farm map; it is not part of the survey record.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema

from herdsight.herd.messages import SCHEMA

ACTIONS = ("second_check", "records_updated", "found_them", "dismissed")

_TABLES = """
CREATE TABLE IF NOT EXISTS farms (farm_id TEXT PRIMARY KEY, layout TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS surveys (
    survey_id TEXT PRIMARY KEY, farm_id TEXT NOT NULL, camp_id TEXT NOT NULL,
    surveyed_at TEXT NOT NULL, record TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS surveys_camp ON surveys (farm_id, camp_id, surveyed_at);
CREATE TABLE IF NOT EXISTS actions (
    alert_id TEXT NOT NULL, action TEXT NOT NULL, at TEXT NOT NULL, note TEXT NOT NULL DEFAULT '');
"""


@lru_cache(maxsize=1)
def _validator() -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8")))


def validate(record: dict[str, Any]) -> None:
    """Raise jsonschema.ValidationError if the record does not match the schema."""
    _validator().validate(record)


class Store:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.executescript(_TABLES)

    def close(self) -> None:
        self.db.close()

    # writing ------------------------------------------------------------------------------
    def set_layout(self, layout: dict[str, Any]) -> None:
        if not layout.get("farm_id") or not isinstance(layout.get("camps"), list):
            raise ValueError("a layout needs farm_id and a list of camps")
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO farms VALUES (?, ?)",
                (layout["farm_id"], json.dumps(layout)),
            )

    def add(self, record: dict[str, Any]) -> None:
        validate(record)
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO surveys VALUES (?, ?, ?, ?, ?)",
                (
                    record["survey_id"],
                    record["farm_id"],
                    record["camp_id"],
                    record["surveyed_at"],
                    json.dumps(record),
                ),
            )

    def act(self, alert_id: str, action: str, note: str = "") -> dict[str, str]:
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}; one of {ACTIONS}")
        if not self._alert_exists(alert_id):
            raise KeyError(alert_id)
        at = datetime.now(UTC).isoformat(timespec="seconds")
        with self.db:
            self.db.execute("INSERT INTO actions VALUES (?, ?, ?, ?)", (alert_id, action, at, note))
        return {"alert_id": alert_id, "action": action, "at": at}

    # reading ------------------------------------------------------------------------------
    def farms(self) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT farm_id FROM farms ORDER BY farm_id")]

    def layout(self, farm_id: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT layout FROM farms WHERE farm_id = ?", (farm_id,)).fetchone()
        return None if row is None else json.loads(row[0])

    def history(self, farm_id: str, camp_id: str, limit: int = 30) -> list[dict[str, Any]]:
        """The camp's records, newest first."""
        rows = self.db.execute(
            "SELECT record FROM surveys WHERE farm_id = ? AND camp_id = ? "
            "ORDER BY surveyed_at DESC LIMIT ?",
            (farm_id, camp_id, limit),
        )
        return [json.loads(r[0]) for r in rows]

    def latest(self, farm_id: str) -> dict[str, dict[str, Any]]:
        """The newest record of every camp that has one."""
        rows = self.db.execute(
            "SELECT s.camp_id, s.record FROM surveys s JOIN ("
            "  SELECT camp_id, MAX(surveyed_at) AS t FROM surveys"
            "  WHERE farm_id = ? GROUP BY camp_id"
            ") m ON s.camp_id = m.camp_id AND s.surveyed_at = m.t WHERE s.farm_id = ?",
            (farm_id, farm_id),
        )
        return {camp: json.loads(rec) for camp, rec in rows}

    def actions(self, alert_id: str) -> list[dict[str, str]]:
        rows = self.db.execute(
            "SELECT action, at, note FROM actions WHERE alert_id = ? ORDER BY at", (alert_id,)
        )
        return [{"action": a, "at": t, "note": n} for a, t, n in rows]

    def open_alerts(self, farm_id: str) -> list[dict[str, Any]]:
        """Alerts on each camp's newest record that the farmer has not closed, newest first.

        An alert is closed by "records_updated", "found_them" or "dismissed"; asking for a second
        check keeps it open until the second check's own record replaces it.
        """
        closing = {"records_updated", "found_them", "dismissed"}
        out = []
        for camp, rec in self.latest(farm_id).items():
            for a in rec["alerts"]:
                done = [x for x in self.actions(a["alert_id"]) if x["action"] in closing]
                if not done:
                    out.append(
                        {
                            **a,
                            "camp_id": camp,
                            "camp_name": rec.get("camp_name", camp),
                            "surveyed_at": rec["surveyed_at"],
                            "status": rec["status"],
                            "actions": self.actions(a["alert_id"]),
                        }
                    )
        return sorted(out, key=lambda a: a["surveyed_at"], reverse=True)

    def _alert_exists(self, alert_id: str) -> bool:
        for (rec,) in self.db.execute("SELECT record FROM surveys"):
            if any(a["alert_id"] == alert_id for a in json.loads(rec)["alerts"]):
                return True
        return False
