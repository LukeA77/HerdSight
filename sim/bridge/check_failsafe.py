"""Judge a link-loss or geofence flight (run_flight.sh) from PX4's log. Simulation only.

Usage (inside the simulation image):
  python3 sim/bridge/check_failsafe.py link|geofence <run folder> <out.json>

Rules (Phase 5 plan, fixed before these flights), on Gazebo's true position as logged by PX4:
  link      PX4 starts its return within 13 s of the link being lost (10 s timeout plus 3 s);
  geofence  PX4 starts its return and the drone never goes more than 165 m from launch (a 150 m
            fence plus 15 m to stop);
  both      the drone stays at least 70 m up until back over launch and lands within 30 m of it.
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
from pyulog import ULog

NAV_RTL = 5


def wind_note(run: str) -> str:
    """The result's note, naming the wind if the flight had one (run_flight.sh writes wind.txt)."""
    w = Path(run) / "wind.txt"
    wind = w.read_text().strip() if w.exists() else "no wind"
    return f"Simulation result, one flight on one world, {wind}; not a field result."


def main() -> int:
    kind, run, dest = sys.argv[1:4]
    log = ULog(sorted(glob.glob(f"{run}/px4_ulog/*/*.ulg"))[-1])

    def topic(name: str) -> dict:
        return next(x for x in log.data_list if x.name == name).data

    q = topic("vehicle_local_position_groundtruth")
    t0 = q["timestamp"][0]
    n, e, h = q["x"], q["y"], -q["z"]
    dist = np.hypot(n, e)
    s = topic("vehicle_status")
    st, nav = (s["timestamp"] - t0) / 1e6, s["nav_state"]
    rtl = np.nonzero(nav == NAV_RTL)[0]
    rtl_t = float(st[rtl[0]]) if len(rtl) else None
    top = int(np.argmax(h > 80))
    far = top + int(np.argmax(dist[top:] > 20))
    back = far + int(np.argmax(dist[far:] < 5))
    lowest = float(h[top:back].min()) if back > top else None
    landed = float(dist[-1]) <= 30 and float(h[-1]) < 1
    out = {
        "kind": kind,
        "note": wind_note(run),
        "return_started_s": rtl_t,
        "lowest_height_until_back_over_launch_m": lowest,
        "largest_distance_from_launch_m": float(dist.max()),
        "touchdown_distance_from_launch_m": float(dist[-1]),
        "final_height_m": float(h[-1]),
    }
    ok = rtl_t is not None and lowest is not None and lowest >= 70 and landed
    if kind == "link":
        # PX4 records when it declares the link lost (the last rise of the flag, since the flag
        # is also set before the ground station first connects); heartbeats stopped 10 s earlier
        flags = topic("failsafe_flags")
        rises = np.nonzero(np.diff(flags["gcs_connection_lost"].astype(int)) == 1)[0]
        if not len(rises):
            raise SystemExit("PX4 never declared the link lost")
        declared = (int(flags["timestamp"][rises[-1] + 1]) - int(t0)) / 1e6
        lost = declared - 10.0
        out["link_lost_s"] = lost
        out["seconds_from_link_loss_to_return"] = None if rtl_t is None else rtl_t - lost
        ok = ok and 0 <= rtl_t - lost <= 13
    else:
        ok = ok and float(dist.max()) <= 165
    out["passes"] = bool(ok)
    Path(dest).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
