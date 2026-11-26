"""Mock farm for building and testing the dashboard before the pipeline is finished (D-17).

Usage: python -m herdsight.dashboard.mock --db <file.sqlite> [--seed 0]

Writes one made-up farm: six camps with outlines on a local grid and eight weekly drone checks per
camp, built with herdsight.herd.messages.survey_record so every record is a real schema record. The
newest checks cover every status the farm map shows: all here, checking again, may have moved, some
may be missing, and not fully checked. Every record is marked simulated; none is a field result.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

import numpy as np

from herdsight.dashboard.store import Store
from herdsight.herd.alerts import Rule
from herdsight.herd.messages import survey_record
from herdsight.herd.sim import Survey

FARM_ID = "demo-farm"
LEVEL = 0.9
WEEKS = 8
# (camp id, name, herd size, corner east, corner north, width, height) in metres
CAMPS = [
    ("c1", "River camp", 48, 0, 0, 420, 300),
    ("c2", "Top camp", 62, 0, 320, 420, 260),
    ("c3", "Dam camp", 35, 440, 0, 300, 300),
    ("c4", "Bush camp", 80, 440, 320, 300, 260),
    ("c5", "Calving camp", 27, 760, 0, 260, 300),
    ("c6", "Hill camp", 54, 760, 320, 260, 260),
]


def layout() -> dict:
    camps = [
        {
            "camp_id": cid,
            "name": name,
            "outline": [[e, n], [e + w, n], [e + w, n + h], [e, n + h]],
        }
        for cid, name, _, e, n, w, h in CAMPS
    ]
    return {"farm_id": FARM_ID, "name": "Demo farm (made-up data)", "camps": camps}


def _survey(
    day: float, camp: str, true: int, count: int, half: float, dense: bool = False
) -> Survey:
    return Survey(day, camp, true, count, {LEVEL: count - half}, {LEVEL: count + half}, dense)


def records(seed: int = 0, start: datetime | None = None) -> list[dict]:
    rng = np.random.default_rng(seed)
    start = start or datetime(2026, 11, 1, tzinfo=UTC)
    rule = Rule("recount", level=LEVEL)
    names = {cid: name for cid, name, *_ in CAMPS}
    out = []
    for cid, _, herd, *_ in CAMPS:
        for week in range(WEEKS):
            day = 7.0 * week
            last = week == WEEKS - 1
            true = herd
            if cid == "c4" and last:
                true = herd - 9  # some animals missing, confirmed by a second check
            if cid == "c3" and last:
                true = herd - 6  # moved to the calving camp
            if cid == "c5" and last:
                true = herd + 6
            half = max(2.0, 0.06 * true)
            count = int(round(true + rng.normal(0, 0.4 * half)))
            s = _survey(day, cid, true, count, half, dense=cid == "c2" and last)
            kw: dict = {}
            if last and cid == "c4":
                kw["alert"] = {"kind": "missing", "checks": 2}
            elif last and cid == "c3":
                kw["alert"] = {"kind": "moved", "checks": 2, "other_camp": "c5"}
            elif last and cid == "c6":
                kw["recount_pending"] = True
                s = _survey(day, cid, herd, herd - 5, half)
            elif last and cid == "c1":
                kw["coverage"] = 0.62
            out.append(
                survey_record(
                    s,
                    farm_id=FARM_ID,
                    expected=herd,
                    rule=rule,
                    camp_names=names,
                    start=start,
                    count_error_model="mock",
                    pipeline_version="mock",
                    **kw,
                )
            )
    return out


def fill(store: Store, seed: int = 0) -> int:
    store.set_layout(layout())
    recs = records(seed)
    for r in recs:
        store.add(r)
    return len(recs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    n = fill(Store(args.db), args.seed)
    print(f"wrote {n} made-up survey records for {FARM_ID} to {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
