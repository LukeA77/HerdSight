"""Check that simulated wind acts on the drone: its lean while holding position. Simulation only.

Usage (inside the simulation image):  python3 sim/bridge/check_wind_tilt.py <run folder> [...]

To hold position in a steady wind a multirotor leans into it by about atan(F / (m g)), where F
is the wind's push. For each run this prints the median tilt (angle of the body z axis from
vertical, from PX4's vehicle_attitude) over the moments the drone is above 70 m and moving
slower than 0.5 m/s horizontally (holding for a still), and how many samples that is, and writes
the same to <run folder>/wind_tilt.json.
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
from pyulog import ULog


def tilt_deg(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    # body z axis in the world frame: third column of the rotation matrix
    zz = 1 - 2 * (x * x + y * y)
    return np.degrees(np.arccos(np.clip(zz, -1, 1)))


def main() -> int:
    for run in sys.argv[1:]:
        f = sorted(glob.glob(f"{run}/px4_ulog/**/*.ulg", recursive=True))[-1]
        u = ULog(f, ["vehicle_attitude", "vehicle_local_position"])
        att = next(d for d in u.data_list if d.name == "vehicle_attitude").data
        pos = next(d for d in u.data_list if d.name == "vehicle_local_position").data
        q = np.vstack([att[f"q[{i}]"] for i in range(4)])
        t_att = att["timestamp"]
        speed = np.hypot(pos["vx"], pos["vy"])
        hold = (-pos["z"] > 70) & (speed < 0.5)
        idx = np.searchsorted(pos["timestamp"], t_att).clip(0, len(hold) - 1)
        sel = hold[idx]
        tilt = tilt_deg(q)[sel]
        med = float(np.median(tilt)) if tilt.size else float("nan")
        print(f"{run}: median tilt while holding {med:.2f} deg over {int(sel.sum())} samples")
        out = {"median_tilt_holding_deg": med, "samples": int(sel.sum())}
        (Path(run) / "wind_tilt.json").write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
