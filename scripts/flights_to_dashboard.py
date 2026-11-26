"""Put finished simulated survey flights on the farmer dashboard (Phase 6 exit: mission -> count ->
dashboard).

Usage: python scripts/flights_to_dashboard.py --db <file.sqlite> --run <name>=<flight folder> ...

Each flight folder is a Gazebo flight from sim/bridge/run_flight.sh with BBOX=1. It is counted at
home base exactly as for the S5 checks (herdsight.sim.home_base.count_run), and its survey result
record is stored as one camp of a simulated farm named "sim-farm". Every flight comes from its own
simulated world, so each is shown as its own camp; the camp's outline is the area the survey
photographed, and camps are laid side by side on the map. The records stay marked simulated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from herdsight.dashboard.store import Store, validate
from herdsight.sim.home_base import count_run

FARM_ID = "sim-farm"
GAP_M = 40.0


def outline(plan: dict, east0: float) -> tuple[list[list[float]], float]:
    """The photographed area as a rectangle, shifted to start at east0; and its width."""
    pts = [p["enu"] for s in plan["sorties"] for p in s["photos"]]
    fw, fh = plan["footprint_m"]
    e = [p[0] for p in pts]
    n = [p[1] for p in pts]
    e_lo, e_hi = min(e) - fw / 2, max(e) + fw / 2
    n_lo, n_hi = min(n) - fh / 2, max(n) + fh / 2
    w = e_hi - e_lo
    rect = [[east0, n_lo], [east0 + w, n_lo], [east0 + w, n_hi], [east0, n_hi]]
    return [[round(x, 1), round(y, 1)] for x, y in rect], w


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--run", action="append", required=True, help="name=flight folder")
    args = ap.parse_args(argv)
    store = Store(args.db)
    camps, east = [], 0.0
    for item in args.run:
        name, path = item.split("=", 1)
        run = Path(path)
        result = count_run(run, farm_id=FARM_ID)
        rec = dict(result["record"])
        rec["camp_id"], rec["camp_name"] = name, f"Simulated camp {name}"
        rec["survey_id"] = f"{FARM_ID}-{name}-{rec['surveyed_at']}"
        for a in rec["alerts"]:
            a["alert_id"] = f"{FARM_ID}-{name}-{a['alert_id']}"
        validate(rec)
        store.add(rec)
        plan = yaml.safe_load((run / "plan.yaml").read_text(encoding="utf-8"))
        rect, width = outline(plan, east)
        east += width + GAP_M
        camps.append({"camp_id": name, "name": rec["camp_name"], "outline": rect})
        print(f"{name}: counted {rec['count']}, records {rec['expected_count']}, {rec['status']}")
    store.set_layout(
        {"farm_id": FARM_ID, "name": "Simulated farm (Gazebo flights)", "camps": camps}
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
