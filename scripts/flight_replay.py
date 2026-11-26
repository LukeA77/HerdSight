"""Replay stored simulated flights through home-base counting and compare with saved results.

Usage:
  python scripts/flight_replay.py check     # against results/phase5/flight_replay_golden.json
  python scripts/flight_replay.py update --note "<why the results changed>"

Regression check (Phase 5): a fixed set of finished Gazebo flights, kept in the storage folder
(runs/phase5_sitl/<name>), is counted again with today's code (herdsight.sim.home_base.count_run).
`check` fails if any count, likely range, status or S5 check differs from the saved results, so an
unintended change to detection, matching, counting or wording is caught. When a change is meant,
`update` rewrites the saved results with a note saying why. Simulation only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from herdsight.sim.home_base import count_run, read_frames
from herdsight.sim.s5_checks import check_flight
from herdsight.utils.config import REPO_ROOT
from herdsight.utils.paths import storage_path

GOLDEN = REPO_ROOT / "results" / "phase5" / "flight_replay_golden.json"
FLIGHTS = {
    "s5_batch_seed3": "complete",
    "s5_batch_seed4": "complete",
    "s5_batch_seed5": "complete",
}


def flight_dir(name: str) -> Path:
    return storage_path("runs", "phase5_sitl", name)


def summarise(name: str, case: str) -> dict[str, Any]:
    run = flight_dir(name)
    result = count_run(run)
    rec = result["record"]
    checks = check_flight(result, read_frames(run / "bbox.jsonl"), case)
    return {
        "count": rec["count"],
        "expected_count": rec["expected_count"],
        "status": rec["status"],
        "intervals": rec["intervals"],
        "notes": rec["notes"],
        "alerts": [a["message"] for a in rec["alerts"]],
        "s5_passes": checks["passes"],
    }


def available() -> list[str]:
    try:
        return [n for n in FLIGHTS if (flight_dir(n) / "bbox.jsonl").exists()]
    except (OSError, ValueError, RuntimeError):  # no storage folder set (for example in CI)
        return []


def check() -> list[str]:
    """Differences from the saved results, one line each (empty when nothing changed)."""
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))["flights"]
    diffs = []
    for name in available():
        now = summarise(name, FLIGHTS[name])
        for key, want in golden[name].items():
            if now.get(key) != want:
                diffs.append(f"{name}: {key} was {want!r}, now {now.get(key)!r}")
    return diffs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("check", "update"))
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)
    if args.step == "update":
        if not args.note:
            ap.error("update needs --note saying why the results changed")
        names = available()
        if len(names) != len(FLIGHTS):
            sys.exit(f"stored flights missing: {sorted(set(FLIGHTS) - set(names))}")
        out = {
            "note": "Simulation: home-base results of stored Gazebo flights; not field results.",
            "last_change": args.note,
            "flights": {n: summarise(n, FLIGHTS[n]) for n in names},
        }
        GOLDEN.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
        print(f"saved results for {len(names)} flights")
        return 0
    if not available():
        print("no stored flights in the storage folder: nothing to replay")
        return 0
    diffs = check()
    print("\n".join(diffs) if diffs else f"{len(available())} flights replayed: no change")
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
