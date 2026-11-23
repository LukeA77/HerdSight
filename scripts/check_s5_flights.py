"""Run the certain S5 checks on finished Gazebo flights (Phase 5 plan, section 8).

Usage: python scripts/check_s5_flights.py --run normal=complete=<dir> --run crash=early=<dir>

Simulation only. One flight per case on one world, so this shows the mission chain works; it is not
an S5 pass rate, and S5 itself also needs the dashboard (D-40).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from herdsight.sim.home_base import count_run, read_frames
from herdsight.sim.s5_checks import check_flight
from herdsight.utils.config import REPO_ROOT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="name=complete|early=directory")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results/phase5/s5_flight_checks.json")
    args = ap.parse_args(argv)
    flights = {}
    for item in args.run:
        name, case, path = item.split("=", 2)
        run = Path(path)
        flights[name] = check_flight(count_run(run), read_frames(run / "bbox.jsonl"), case)
    out = {
        "note": (
            "Simulation results, one flight per case on one world, no wind, ideal stabilised "
            "camera; not field results and not an S5 pass rate."
        ),
        "all_pass": all(f["passes"] for f in flights.values()),
        "flights": flights,
    }
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
