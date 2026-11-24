"""Summarise a GNSS-loss flight (run_flight.sh with INJECT=gnss) from PX4's log. Simulation only.

Usage (inside the simulation image): python3 sim/bridge/check_gnss_loss.py <run folder> <out.json>

Uses Gazebo's true position as logged by PX4, since PX4's own estimate is what fails. Distance to
the herd uses the animals' positions at the start of the flight, so it is indicative only.
"""

import glob
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from pyulog import ULog

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

# make_world.py defaults since D-46; flights before it used a camp from 20 to 270 m east
CAMP_EAST_M, CAMP_NORTH_HALF_M = (35.0, 285.0), 90.0


def wind_note(run: str) -> str:
    """The result's note, naming the wind if the flight had one (run_flight.sh writes wind.txt)."""
    w = Path(run) / "wind.txt"
    wind = w.read_text().strip() if w.exists() else "no wind"
    return f"Simulation result, one flight on one world, {wind}; not a field result."


def main() -> int:
    run, dest = sys.argv[1:3]
    log = ULog(sorted(glob.glob(f"{run}/px4_ulog/*/*.ulg"))[-1])

    def topic(name: str) -> dict:
        return next(x for x in log.data_list if x.name == name).data

    q = topic("vehicle_local_position_groundtruth")
    t0 = q["timestamp"][0]
    t, n, e, h = (q["timestamp"] - t0) / 1e6, q["x"], q["y"], -q["z"]
    gps = topic("sensor_gps")
    lost_t = float((gps["timestamp"][np.argmax(gps["fix_type"] < 3)] - t0) / 1e6)
    k = int(np.searchsorted(t, lost_t))
    status = topic("vehicle_status")
    change = np.nonzero(np.diff(status["nav_state"]))[0]
    states = [
        (float((status["timestamp"][j + 1] - t0) / 1e6), int(status["nav_state"][j + 1]))
        for j in change
    ]
    blind_t = next(s for s, v in states if s > lost_t)
    with open(f"{run}/herd_state.pkl", "rb") as fh:
        herd = pickle.load(fh)
    near = float(np.min(np.hypot(herd.pos[:, 0] - e[-1], herd.pos[:, 1] - n[-1])))
    outside = max(
        abs(float(n[-1])) - CAMP_NORTH_HALF_M,
        CAMP_EAST_M[0] - float(e[-1]),
        float(e[-1]) - CAMP_EAST_M[1],
        0.0,
    )
    out = {
        "note": wind_note(run),
        "seconds_from_gnss_loss_to_px4_failsafe": blind_t - lost_t,
        "px4_action": "blind descent (no position estimate)",
        "seconds_from_gnss_loss_to_touchdown": float(t[-1] - lost_t),
        "height_at_gnss_loss_m": float(h[k]),
        "horizontal_travel_from_loss_to_touchdown_m": float(np.hypot(n[-1] - n[k], e[-1] - e[k])),
        "touchdown_distance_from_launch_m": float(np.hypot(n[-1], e[-1])),
        "touchdown_outside_camp_m": outside,
        "touchdown_to_nearest_animal_start_position_m": near,
        "animals": int(len(herd.pos)),
    }
    Path(dest).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
