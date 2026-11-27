"""Score the S6 usability sessions from the filled-in recording sheets (Phase 6).

Usage: python scripts/score_s6.py <sessions.csv>

The CSV has one row per tester, with the columns of the recording sheet in
docs/plans/s6-usability-test.md:
  tester, date, task1_camp, task1_seconds, task2_said, task3_without_help, task3_seconds,
  task4_camps, task5_worked_offline, confusing, trust
task3_without_help and task5_worked_offline are yes or no. Keep the CSV private (it may hold
names or remarks); this script writes only initials, times, the pass result and the remarks as
typed, so type remarks without names. The pass rule is the one fixed before any session: Task 1
names Bush camp within 10 seconds, and Task 3 is done without help within 60 seconds. Writes
results/phase6/s6_usability.json.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from herdsight.utils.config import REPO_ROOT, load_config

URGENT_CAMP = "bush"
TASK1_MAX_S, TASK3_MAX_S = 10.0, 60.0


def yes(v: str) -> bool:
    return v.strip().lower() in ("yes", "y", "true", "1")


def initials(name: str) -> str:
    parts = [p for p in name.replace(".", " ").split() if p]
    return "".join(p[0].upper() for p in parts) if len(parts) > 1 else name.strip()[:3].upper()


def score_row(r: dict) -> dict:
    t1 = float(r["task1_seconds"])
    t3 = float(r["task3_seconds"]) if r["task3_seconds"].strip() else None
    task1 = URGENT_CAMP in r["task1_camp"].lower() and t1 <= TASK1_MAX_S
    task3 = yes(r["task3_without_help"]) and t3 is not None and t3 <= TASK3_MAX_S
    return {
        "tester": initials(r["tester"]),
        "date": r["date"].strip(),
        "task1": {"camp": r["task1_camp"].strip(), "seconds": t1, "pass": task1},
        "task2_said": r["task2_said"].strip(),
        "task3": {"without_help": yes(r["task3_without_help"]), "seconds": t3, "pass": task3},
        "task4_camps": r["task4_camps"].strip(),
        "task5_worked_offline": yes(r["task5_worked_offline"]),
        "confusing": r["confusing"].strip(),
        "trust": r["trust"].strip(),
        "s6_pass": bool(task1 and task3),
    }


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print(__doc__)
        return 2
    with Path(args[0]).open(encoding="utf-8-sig", newline="") as fh:
        rows = [score_row(r) for r in csv.DictReader(fh) if r.get("tester", "").strip()]
    need = int(load_config("specs").require("s6_min_testers"))
    passed = sum(r["s6_pass"] for r in rows)
    out = {
        "note": "Usability sessions with non-technical testers on the made-up farm (not real data)",
        "rule": (
            f"a tester passes if Task 1 names Bush camp within {TASK1_MAX_S:g} s and Task 3 is "
            f"done without help within {TASK3_MAX_S:g} s (docs/plans/s6-usability-test.md)"
        ),
        "testers": len(rows),
        "testers_needed": need,
        "passed": passed,
        "summary": f"{passed} of {len(rows)} testers passed",
        "sessions": rows,
    }
    dest = REPO_ROOT / "results" / "phase6" / "s6_usability.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(out["summary"], "->", dest)
    if len(rows) < need:
        print(f"note: {need} testers are planned; {len(rows)} entered so far")
    return 0


if __name__ == "__main__":
    sys.exit(main())
