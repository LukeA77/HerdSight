"""Record the bounding-box cameras during a simulated flight (Phase 5, step 8).

Usage (inside the simulation image): python3 record_bbox.py --out bbox.jsonl

One line per frame: sensor, wall-clock receive time, simulation time and each labelled box in
pixels. Frames arrive late when rendering falls behind, so their receive time must not be used to
pair them with stills. `clock` lines (10 per second) record the simulation time at a wall-clock
time; they convert a still's wall-clock trigger time to simulation time, and frames are then
paired by simulation time. `pose` lines hold the true drone and camera positions and `herd`
lines (once a second) every animal's true position, for the welfare check. Simulation only.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from gz.msgs10.annotated_axis_aligned_2d_box_v_pb2 import AnnotatedAxisAligned2DBox_V
from gz.msgs10.clock_pb2 import Clock
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node

TOPICS = {"visible": "/survey_bbox_visible", "full": "/survey_bbox_full"}
CLOCK_PERIOD_S = 0.1
HERD_PERIOD_S = 1.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--world", default="herdsight_farm")
    args = ap.parse_args(argv)
    node = Node()
    lock = threading.Lock()
    stop = threading.Event()
    with args.out.open("w", encoding="utf-8") as fh:

        def callback(kind: str) -> Callable[[AnnotatedAxisAligned2DBox_V], None]:
            def on_frame(msg: AnnotatedAxisAligned2DBox_V) -> None:
                row = {
                    "sensor": kind,
                    "wall_s": time.time(),
                    "sim_s": msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9,
                    "boxes": [
                        [
                            b.label,
                            b.box.min_corner.x,
                            b.box.min_corner.y,
                            b.box.max_corner.x,
                            b.box.max_corner.y,
                        ]
                        for b in msg.annotated_box
                    ],
                }
                with lock:
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()

            return on_frame

        last_clock = {"wall_s": 0.0}

        def on_clock(msg: Clock) -> None:
            now = time.time()
            if now - last_clock["wall_s"] < CLOCK_PERIOD_S:
                return
            last_clock["wall_s"] = now
            row = {"sensor": "clock", "wall_s": now, "sim_s": msg.sim.sec + msg.sim.nsec * 1e-9}
            with lock:
                fh.write(json.dumps(row) + "\n")
                fh.flush()

        last_pose = {"wall_s": 0.0}
        last_herd = {"wall_s": 0.0}

        def on_poses(msg: Pose_V) -> None:
            now = time.time()
            sim_s = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
            if now - last_herd["wall_s"] >= HERD_PERIOD_S:
                animals = [
                    (int(p.name.split("_")[1]), p.position.x, p.position.y)
                    for p in msg.pose
                    if p.name.startswith("animal_")
                ]
                if animals:
                    last_herd["wall_s"] = now
                    xy = [[x, y] for _, x, y in sorted(animals)]
                    with lock:
                        fh.write(
                            json.dumps({"sensor": "herd", "wall_s": now, "sim_s": sim_s, "xy": xy})
                            + "\n"
                        )
                        fh.flush()
            if now - last_pose["wall_s"] < CLOCK_PERIOD_S:
                return
            found = {
                p.name: [p.position.x, p.position.y, p.position.z]
                for p in msg.pose
                if p.name in ("x500_0", "survey_camera")
            }
            if len(found) < 2:
                return
            last_pose["wall_s"] = now
            row = {
                "sensor": "pose",
                "wall_s": now,
                "sim_s": sim_s,
                "drone": found["x500_0"],
                "camera": found["survey_camera"],
            }
            with lock:
                fh.write(json.dumps(row) + "\n")
                fh.flush()

        if not node.subscribe(Pose_V, f"/world/{args.world}/pose/info", on_poses):
            print("could not subscribe to the world's poses", file=sys.stderr)
            return 1
        clock_topic = f"/world/{args.world}/clock"
        if not node.subscribe(Clock, clock_topic, on_clock):
            print(f"could not subscribe to {clock_topic}", file=sys.stderr)
            return 1
        for kind, topic in TOPICS.items():
            if not node.subscribe(AnnotatedAxisAligned2DBox_V, topic, callback(kind)):
                print(f"could not subscribe to {topic}", file=sys.stderr)
                return 1
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        stop.wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
