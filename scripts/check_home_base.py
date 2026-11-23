"""Count finished Gazebo flights at home base and write the check (Phase 5, step 8).

Usage: python scripts/check_home_base.py --run normal=<dir> --run herd=<dir> --run crash=<dir>

Each flight is counted with many draws of the detection error model, because one draw says little:
the flight is fixed, and the draws show how much the count varies with detection and geotag error.
One flight per case on one world, no wind, with an ideal stabilised camera. Simulation only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from herdsight.sim.home_base import count_run
from herdsight.utils.config import REPO_ROOT

N_DRAWS = 50


def check_flight(run: Path, draws: int = N_DRAWS) -> dict:
    results = [count_run(run, seed=k) for k in range(draws)]
    first = results[0]
    counts = np.array([r["count"]["count"] for r in results])
    statuses = [r["record"]["status"] for r in results]
    alerts = sum(bool(r["record"]["alerts"]) for r in results)
    true = int(first["checks"]["animals_in_world"])
    bundle = first["bundle"]
    return {
        "outcome": bundle["outcome"],
        "photos_planned": bundle["photos_planned"],
        "photos_taken": bundle["photos_taken"],
        "herd_mode_stills": bundle["herd_mode_stills"],
        "coverage": bundle["coverage"],
        "true_animals": true,
        "draws": draws,
        "count_median": float(np.median(counts)),
        "count_min": int(counts.min()),
        "count_max": int(counts.max()),
        "mean_absolute_relative_error": float(np.mean(np.abs(counts - true)) / true),
        "status_counts": {s: statuses.count(s) for s in sorted(set(statuses))},
        "draws_with_an_alert": alerts,
        "all_records_marked_simulated": all(r["record"]["simulated"] for r in results),
        "checks": first["checks"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="name=directory")
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results/phase5/home_base_check.json")
    args = ap.parse_args(argv)
    flights = {}
    for item in args.run:
        name, path = item.split("=", 1)
        flights[name] = check_flight(Path(path), args.draws)
    out = {
        "note": (
            "Simulation results, one flight per case on one world, no wind, ideal stabilised "
            "camera; not field results. Spread is over draws of the detection error model on "
            "the same flight, so it is not the error across flights or farms."
        ),
        "flights": flights,
    }
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
