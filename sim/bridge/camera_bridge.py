"""Keep the stabilised survey camera over the drone during a flight (Phase 5, step 8).

Usage (inside the simulation image): python3 camera_bridge.py --plan plan.yaml

Reads the drone's true pose from Gazebo and moves the camera model (sim/bridge/add_survey_camera.py)
to it, pointing straight down. Gazebo applies pose commands more slowly than they can be sent,
and a backlog leaves the camera far behind, so a new command goes out only once the last one has
taken effect (or after a short timeout). Simulation only.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from add_survey_camera import MODEL_NAME, nadir_quaternion  # noqa: E402
from gz.msgs10.boolean_pb2 import Boolean  # noqa: E402
from gz.msgs10.pose_pb2 import Pose  # noqa: E402
from gz.msgs10.pose_v_pb2 import Pose_V  # noqa: E402
from gz.transport13 import Node  # noqa: E402

PERIOD_S = 0.01
APPLIED_WITHIN_M = 0.02
RESEND_AFTER_S = 1.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--world", default="herdsight_farm")
    ap.add_argument("--drone", default="x500", help="model name prefix of the drone")
    args = ap.parse_args(argv)
    plan = yaml.safe_load(args.plan.read_text(encoding="utf-8"))
    w, x, y, z = nadir_quaternion(float(plan["heading_enu_rad"]))
    node = Node()
    drone: dict[str, tuple[float, float, float] | None] = {"xyz": None, "camera": None}

    def on_poses(msg: Pose_V) -> None:
        for p in msg.pose:
            if p.name == MODEL_NAME:
                drone["camera"] = (p.position.x, p.position.y, p.position.z)
            elif p.name.startswith(args.drone):
                drone["xyz"] = (p.position.x, p.position.y, p.position.z)

    if not node.subscribe(Pose_V, f"/world/{args.world}/pose/info", on_poses):
        print("could not subscribe to the world's poses", file=sys.stderr)
        return 1
    service = f"/world/{args.world}/set_pose"
    sent: tuple[float, float, float] | None = None
    sent_at = 0.0
    while True:
        t0 = time.monotonic()
        xyz, camera = drone["xyz"], drone["camera"]
        applied = (
            sent is None
            or (camera is not None and math.dist(camera, sent) <= APPLIED_WITHIN_M)
            or t0 - sent_at >= RESEND_AFTER_S
        )
        if xyz is not None and applied:
            sent, sent_at = xyz, t0
            req = Pose()
            req.name = MODEL_NAME
            req.position.x, req.position.y, req.position.z = xyz
            req.orientation.w, req.orientation.x = w, x
            req.orientation.y, req.orientation.z = y, z
            node.request(service, req, Pose, Boolean, 200)
        time.sleep(max(0.0, PERIOD_S - (time.monotonic() - t0)))


if __name__ == "__main__":
    sys.exit(main())
