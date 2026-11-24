"""Phase 5: the 145 m geofence trigger (D-47) in wind, judged by the rule fixed before the flights.

Usage: python scripts/phase5_geofence_wind.py   (after the flights and their checks)

Reads each flight's geofence_check.json (sim/bridge/check_failsafe.py) and wind_tilt.json
(sim/bridge/check_wind_tilt.py) from the storage drive and writes
results/phase5/geofence_wind_check.json. Rule, fixed before any wind flight: three flights with
the trigger at 145 m (A: 5 m/s blowing east, B: 10 m/s blowing east, C: 10 m/s blowing straight
outward from launch at the fence crossing, bearing 116 degrees); each must pass the unchanged
geofence rule (return starts, never beyond 165 m, at least 70 m up until back over launch, landing
within 30 m of launch). A flight counts only if the wind acted: the drone's median lean while
holding for a still exceeds the no-wind flight's by at least half the lean expected from the
wind's push, atan(F / (m g)). D-47 is adopted only if A, B and C all pass. Simulation results,
one flight per case on one world, with an assumed drag area; not field results.
"""

from __future__ import annotations

import json
import math

from herdsight.utils.config import REPO_ROOT
from herdsight.utils.paths import storage_path

RHO, CDA, MASS_TOTAL, MASS_BODY, G = 1.225, 0.06, 2.0 + 4 * 0.016076923, 2.0, 9.81
CASES = {
    "no_wind": ("s5_geofence145_launch35", 0.0, 0.0),
    "A_5mps_east": ("geofence145_wind5_east", 5.0, 0.0),
    "B_10mps_east": ("geofence145_wind10_east", 10.0, 0.0),
    "C_10mps_outward_116deg": ("geofence145_wind10_outward", 8.988, -4.384),
}


def expected_lean_deg(speed: float) -> float:
    force = 0.5 * RHO * CDA * speed**2  # what add_wind.py sets for a drone holding position
    return math.degrees(math.atan(force / (MASS_TOTAL * G)))


def main() -> int:
    root = storage_path("runs", "phase5_sitl")
    rows = {}
    for name, (folder, east, north) in CASES.items():
        run = root / folder
        check = json.loads((run / "geofence_check.json").read_text())
        tilt = json.loads((run / "wind_tilt.json").read_text())
        rows[name] = {
            "run": folder,
            "wind_east_mps": east,
            "wind_north_mps": north,
            "wind_speed_mps": round(math.hypot(east, north), 3),
            **{k: check[k] for k in check if k not in ("kind", "note")},
            "median_lean_holding_deg": tilt["median_tilt_holding_deg"],
        }
    base = rows["no_wind"]["median_lean_holding_deg"]
    for name, r in rows.items():
        if name == "no_wind":
            continue
        exp = expected_lean_deg(r["wind_speed_mps"])
        r["expected_lean_deg"] = exp
        r["wind_acted"] = r["median_lean_holding_deg"] - base >= exp / 2
        r["counts_and_passes"] = bool(r["wind_acted"] and r["passes"])
    adopted = all(r["counts_and_passes"] for n, r in rows.items() if n != "no_wind")
    out = {
        "note": (
            "Simulation results, one flight per case on one world, steady wind with an assumed "
            f"drag area of {CDA} m^2 (sim/bridge/add_wind.py); not field results."
        ),
        "trigger_m": 145,
        "limit_m": 165,
        "cases": rows,
        "d47_adopted_by_rule": adopted,
    }
    dest = REPO_ROOT / "results" / "phase5" / "geofence_wind_check.json"
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for n, r in rows.items():
        print(
            n,
            round(r["largest_distance_from_launch_m"], 2),
            round(r["median_lean_holding_deg"], 2),
            r.get("counts_and_passes", "-"),
        )
    print("D-47 adopted by rule:", adopted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
