"""Summarise a camera-return flight (run_flight.sh with VISION=1 INJECT=gnss). Simulation only.

Usage (inside the simulation image): python3 sim/bridge/check_camera_return.py <run> <out.json>

Scores against the rule fixed before the flights (_private notes, 24 Nov): no blind descent,
touchdown within 10 m of launch, and camera fixes for at least 80% of frames between GPS loss and
5 m height with median error <= 3 m and none over 20 m. Truth is Gazebo's position: PX4's logged
ground truth for the touchdown, and the camera node's log for the fixes (its true positions are in
the world frame, so the offset to PX4's frame is the median difference while GPS was good).
"""

import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
from pyulog import ULog

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_gnss_loss import wind_note  # noqa: E402

DESCEND = 12  # vehicle_status nav_state: descend without a position estimate
NAMES = {
    2: "position hold",
    3: "mission",
    4: "loiter",
    5: "return to launch",
    12: "blind descent",
    14: "offboard",
    18: "land",
    17: "take-off",
}


def main() -> int:
    run, dest = sys.argv[1:3]
    log = ULog(sorted(glob.glob(f"{run}/px4_ulog/*/*.ulg"))[-1])

    def topic(name: str) -> dict:
        return next(x for x in log.data_list if x.name == name).data

    q = topic("vehicle_local_position_groundtruth")
    t0 = q["timestamp"][0]
    t, n, e = (q["timestamp"] - t0) / 1e6, q["x"], q["y"]
    gps = topic("sensor_gps")
    lost_t = float((gps["timestamp"][np.argmax(gps["fix_type"] < 3)] - t0) / 1e6)
    status = topic("vehicle_status")
    after = status["timestamp"] - t0 > lost_t * 1e6
    states = [int(v) for v in status["nav_state"][after]]
    seen = list(dict.fromkeys(states))
    landed = float(np.hypot(n[-1], e[-1]))

    rows = list(csv.DictReader(open(f"{run}/camera_return.csv")))
    good = [r for r in rows if r["event"] == "keyframe" and r["true_e"]]
    off = np.median(
        [
            [float(r["true_e"]) - float(r["est_e"]), float(r["true_n"]) - float(r["est_n"])]
            for r in good
        ],
        axis=0,
    )
    frames = [r for r in rows if r["event"] in ("fix", "no_fix") and float(r["height"]) >= 5]
    fixes = [r for r in frames if r["event"] == "fix" and r["true_e"]]
    err = np.array(
        [
            np.hypot(
                float(r["fix_e"]) + off[0] - float(r["true_e"]),
                float(r["fix_n"]) + off[1] - float(r["true_n"]),
            )
            for r in fixes
        ]
    )
    share = len(fixes) / max(len(frames), 1)
    # in flight only (5 m and up, as for the fixes): after touchdown the node keeps logging the
    # parked drone thousands of times, which swamped this measure before 25 Nov
    vo = [
        r
        for r in rows
        if r["event"] == "vo" and r["true_e"] and r["fix_e"] and float(r["height"]) >= 5
    ]
    vo_err = np.array(
        [
            np.hypot(
                float(r["fix_e"]) + off[0] - float(r["true_e"]),
                float(r["fix_n"]) + off[1] - float(r["true_n"]),
            )
            for r in vo
        ]
    )
    out = {
        "note": wind_note(run) + " Camera return home (VISION=1).",
        "keyframes": len(good),
        "px4_modes_after_gnss_loss": [NAMES.get(s, str(s)) for s in seen],
        "blind_descent": DESCEND in seen,
        "touchdown_distance_from_launch_m": landed,
        "frames_after_loss_above_5m": len(frames),
        "fixed_share": share,
        "fix_error_median_m": float(np.median(err)) if len(err) else None,
        "fix_error_max_m": float(np.max(err)) if len(err) else None,
        "seconds_from_gnss_loss_to_touchdown": float(t[-1] - lost_t),
        "position_sent_error_median_m": float(np.median(vo_err)) if len(vo_err) else None,
        "position_sent_error_p90_m": float(np.percentile(vo_err, 90)) if len(vo_err) else None,
    }
    # a flight that never flew a return to launch after GPS loss is invalid, not a pass (test 5)
    out["valid"] = bool("return to launch" in out["px4_modes_after_gnss_loss"] and len(frames))
    sent_med, sent_p90 = out["position_sent_error_median_m"], out["position_sent_error_p90_m"]
    # rule of tests 4 and 5: the position sent to PX4 replaces the share of frames with a fix
    out["passes_test4_rule"] = bool(
        out["valid"]
        and not out["blind_descent"]
        and landed <= 10
        and sent_med is not None
        and sent_med <= 3
        and sent_p90 <= 10
        and len(err)
        and np.median(err) <= 3
        and np.max(err) <= 20
    )
    out["passes"] = bool(
        out["valid"]
        and not out["blind_descent"]
        and landed <= 10
        and share >= 0.8
        and len(err)
        and np.median(err) <= 3
        and np.max(err) <= 20
    )
    Path(dest).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
