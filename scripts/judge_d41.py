"""Judge the D-41 comparison of PX4 interfaces by the rule fixed before the flights.

Usage: python scripts/judge_d41.py [--runs <folder>]

Reads result.json of the six flights (sim/bridge/run_interface_test.sh) from
$HERDSIGHT_STORAGE/runs/phase5_sitl/d41_v2 and applies the rule in
docs/plans/phase5-simulation-mission.md. Whether an interface can insert herd-mode targets in
flight is a property of the interface, recorded here, and was not flown in this comparison.
Writes results/phase5/d41_interfaces.json. One flight per case on one simulated world: this
chooses an interface and is not a reliability estimate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from herdsight.utils.config import REPO_ROOT
from herdsight.utils.paths import storage_path

METHODS = ("custom", "offboard", "mission")
INSERTS_TARGETS = {
    "custom": "yes: the mode chooses its next target each cycle",
    "offboard": "yes: the companion chooses its next setpoint each cycle",
    "mission": "no: a new target needs a mission re-upload",
}
STILL_LIMIT_M, HEIGHT_LIMIT_M, CONTROL_S, MIN_HEIGHT_M, LAND_M = 3.0, 3.0, 3.0, 70.0, 30.0
TIE_SHARE = 0.05


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=None)
    args = ap.parse_args(argv)
    runs = args.runs or storage_path("runs", "phase5_sitl", "d41_v2")
    planned = float(
        yaml.safe_load((runs / "custom_normal" / "plan.yaml").read_text())["height_agl_m"]
    )
    rows = {}
    for m in METHODS:
        n = json.loads((runs / f"{m}_normal" / "result.json").read_text())
        c = json.loads((runs / f"{m}_crash" / "result.json").read_text())
        heights = (n["height_m_min"], n["height_m_max"])
        normal_ok = (
            n["stills"] == n["planned_stills"]
            and n["distance_to_planned_m_max"] <= STILL_LIMIT_M
            and all(abs(h - planned) <= HEIGHT_LIMIT_M for h in heights)
        )
        # an uploaded mission is PX4's own: it simply continues when the companion is lost
        in_control = m == "mission" or (
            c["first_state_change_after_crash_s"] is not None
            and c["first_state_change_after_crash_s"] <= CONTROL_S
        )
        loss_ok = (
            in_control
            and c["lowest_height_after_crash_m"] is not None
            and c["lowest_height_after_crash_m"] >= MIN_HEIGHT_M
            and c["landed_and_disarmed"]
            and c["final_distance_from_launch_m"] <= LAND_M
        )
        inserts = INSERTS_TARGETS[m].startswith("yes")
        rows[m] = {
            "normal": {
                k: n[k]
                for k in (
                    "stills",
                    "planned_stills",
                    "distance_to_planned_m_median",
                    "distance_to_planned_m_max",
                    "height_m_min",
                    "height_m_max",
                    "first_to_last_still_s",
                    "final_distance_from_launch_m",
                )
            },
            "companion_lost_after_22_stills": {
                k: c[k]
                for k in (
                    "first_state_change_after_crash_s",
                    "lowest_height_after_crash_m",
                    "landed_and_disarmed",
                    "final_distance_from_launch_m",
                )
            },
            "inserts_herd_targets": INSERTS_TARGETS[m],
            "criteria": {
                "normal_flight": bool(normal_ok),
                "companion_loss": bool(loss_ok),
                "inserts_targets": inserts,
            },
            "passes": bool(normal_ok and loss_ok and inserts),
        }
    passing = [m for m in METHODS if rows[m]["passes"]]
    time_of = {m: rows[m]["normal"]["first_to_last_still_s"] for m in passing}
    winner = min(time_of, key=time_of.get) if time_of else None
    if winner and "custom" in time_of and time_of["custom"] <= time_of[winner] * (1 + TIE_SHARE):
        winner = "custom"
    out = {
        "note": "Simulation results, one flight per case on one world; not field results.",
        "planned_height_m": planned,
        "interfaces": rows,
        "passing": passing,
        "winner_by_the_fixed_rule": winner,
        "custom_time_over_winner": time_of["custom"] / time_of[winner] - 1
        if winner and "custom" in time_of
        else None,
    }
    dest = REPO_ROOT / "results" / "phase5" / "d41_interfaces.json"
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    for m in METHODS:
        print(m, rows[m]["criteria"], "passes:", rows[m]["passes"], time_of.get(m))
    print("winner:", winner, "custom slower by", out["custom_time_over_winner"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
