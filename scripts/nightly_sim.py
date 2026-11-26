"""Many-seed simulated mission run with a pass rate (Phase 5: the nightly job).

Usage: python scripts/nightly_sim.py --seeds 20 21 22 [--wind 0 5] [--minutes 20] [--name nightly]

Flies one survey per (world seed, wind) with sim/bridge/run_flight.sh in the simulation image
(BBOX=1, no failure injected), one flight at a time, then checks each finished flight with the S5
checks (herdsight.sim.s5_checks.check_flight) and counts it at home base. Writes
results/phase5/<name>.json: every flight's checks and count, and the pass rate with a 95% Wilson
interval. A flight that never finishes (simulator fault) is reported as such, not silently dropped.
Simulation only: an S5 pass rate in simulation, not a field result. Needs Docker and the
HERDSIGHT_STORAGE folder; run overnight, it takes about 15 to 20 minutes per flight.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from herdsight.sim.home_base import count_run, read_frames
from herdsight.sim.s5_checks import check_flight
from herdsight.utils.config import REPO_ROOT
from herdsight.utils.paths import storage_path

IMAGE = "herdsight-sim:dev"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def fly(out: Path, seed: int, wind: float, minutes: int) -> int:
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        "docker", "run", "--rm", "--name", f"hs_nightly_{seed}_{int(wind)}",
        "-e", "BBOX=1", "-e", f"WIND_E={wind:g}",
        "-v", f"{REPO_ROOT.as_posix()}:/repo:ro", "-v", "hs_ros_ws:/ws",
        "-v", f"{out.as_posix()}:/out", IMAGE,
        "bash", "/repo/sim/bridge/run_flight.sh", str(minutes * 60), str(seed),
    ]  # fmt: skip
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    with (out / "console.txt").open("w", encoding="utf-8") as log:
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode


def judge(run: Path) -> dict:
    if not (run / "bbox.jsonl").exists() or not (run / "stills.csv").exists():
        return {"finished": False, "passes": False, "reason": "flight did not produce its records"}
    try:
        result = count_run(run)
        checks = check_flight(result, read_frames(run / "bbox.jsonl"), "complete")
    except Exception as e:  # noqa: BLE001 - a broken flight is reported, not fatal
        return {"finished": False, "passes": False, "reason": f"{type(e).__name__}: {e}"}
    rec = result["record"]
    return {
        "finished": True,
        "passes": bool(checks["passes"]),
        "failed_checks": [k for k, v in checks.get("checks", {}).items() if v is False],
        "count": rec["count"],
        "expected_count": rec["expected_count"],
        "status": rec["status"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--wind", type=float, nargs="+", default=[0.0])
    ap.add_argument("--minutes", type=int, default=20, help="time limit per flight")
    ap.add_argument("--name", default="nightly")
    ap.add_argument("--judge-only", action="store_true", help="re-check flights already flown")
    args = ap.parse_args(argv)
    root = storage_path("runs", "phase5_sitl", args.name, create=True)
    flights = {}
    for wind in args.wind:
        for seed in args.seeds:
            key = f"seed{seed}_wind{wind:g}"
            run = root / key
            t = time.time()
            if not args.judge_only:
                fly(run, seed, wind, args.minutes)
            flights[key] = {**judge(run), "minutes": round((time.time() - t) / 60, 1)}
            print(key, flights[key], flush=True)
    n = len(flights)
    k = sum(f["passes"] for f in flights.values())
    out = {
        "note": "Simulation: S5 checks on simulated survey flights (ideal stabilised camera); "
        "not field results.",
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "flights": flights,
        "passed": k,
        "flown": n,
        "finished": sum(f["finished"] for f in flights.values()),
        "pass_rate": k / n if n else None,
        "pass_rate_ci95": wilson(k, n),
    }
    dest = REPO_ROOT / "results" / "phase5" / f"{args.name}.json"
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"{k} of {n} flights passed; interval {out['pass_rate_ci95']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
