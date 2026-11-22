"""Move the Gazebo cattle with the Python herd model during a flight (Phase 5 plan, step 6).

Usage (inside the simulation image, with the world from python/herdsight/sim/gazebo.py running):
  python3 sim/bridge/herd_bridge.py --world herdsight_farm --state herd_state.npz [--rate 2]

One implementation of herd movement (plan review, finding 20): the Python model in
python/herdsight/sim/herds.py steps in time with the simulation, reads where the drone is (so
animals can react to it), and sets every animal's pose in one call to Gazebo's set_pose_vector
service. The herd it starts from is the one the world was generated from (`--state`, written by
sim/bridge/make_world.py), so the rendered animals and the herd model agree.
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from gz.msgs10.boolean_pb2 import Boolean  # noqa: E402
from gz.msgs10.pose_v_pb2 import Pose_V  # noqa: E402
from gz.transport13 import Node  # noqa: E402


def yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    return math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="herdsight_farm")
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--rate", type=float, default=2.0, help="herd steps per second")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drone", default="x500", help="model name prefix of the drone")
    args = ap.parse_args(argv)
    with args.state.open("rb") as fh:
        herd = pickle.load(fh)  # herdsight.sim.herds.Herd
    rng = np.random.default_rng(args.seed)
    node = Node()
    drone = {"xyz": None}

    def on_poses(msg: Pose_V) -> None:
        for p in msg.pose:
            if p.name.startswith(args.drone):
                drone["xyz"] = (p.position.x, p.position.y, p.position.z)

    node.subscribe(Pose_V, f"/world/{args.world}/pose/info", on_poses)
    service = f"/world/{args.world}/set_pose_vector"
    heading = rng.uniform(-math.pi, math.pi, herd.n)
    dt = 1.0 / args.rate
    while True:
        t0 = time.monotonic()
        old = herd.pos.copy()
        herd.step(dt, rng, drone=drone["xyz"])
        moved = herd.pos - old
        speed = np.linalg.norm(moved, axis=1)
        heading = np.where(speed > 0.05, np.arctan2(moved[:, 1], moved[:, 0]), heading)
        req = Pose_V()
        for i, (x, y) in enumerate(herd.pos):
            p = req.pose.add()
            p.name = f"animal_{i}"
            p.position.x, p.position.y, p.position.z = float(x), float(y), 0.0
            w, qx, qy, qz = yaw_quaternion(float(heading[i]))
            p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z = w, qx, qy, qz
        node.request(service, req, Pose_V, Boolean, 1000)
        time.sleep(max(0.0, dt - (time.monotonic() - t0)))


if __name__ == "__main__":
    sys.exit(main())
