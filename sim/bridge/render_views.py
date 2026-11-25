"""Render downward camera frames at given positions over the simulated farm. Simulation only.

Usage (inside the simulation image):
  python3 sim/bridge/render_views.py --world <world.sdf> --poses <poses.json> --out <folder>
      [--sun-dir X Y Z] [--sun-diffuse R G B]

Adds a free-floating camera (no gravity) to a copy of the world, starts Gazebo headless, then for
each pose {"east", "north", "height", "heading"} (metres, ENU; heading clockwise from north, in
radians) moves the camera there with the world's set_pose service, waits for a frame rendered
after the move, and saves it as <out>/<index>.png. The camera matches the survey camera's footprint
at 83 m (horizontal field of view 1.274 rad, 1280 x 960). --sun-dir and --sun-diffuse change the
light, so teach and repeat frames can be rendered under different lighting. --thermal renders
with a thermal camera instead (640 x 512, horizontal field of view 50 degrees, as a FLIR Boson 640)
and saves 16-bit images in hundredths of a kelvin; run sim/bridge/add_thermal.py on the world
first. Writes <out>/frames.json with each frame's pose.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.image_pb2 import Image as ImgMsg
from gz.msgs10.pose_pb2 import Pose
from gz.transport13 import Node
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from add_survey_camera import nadir_quaternion  # noqa: E402

WORLD = "herdsight_farm"
CAMERA = """    <model name="view_camera"><pose>0 0 83 0 1.5707963 1.5707963</pose>
      <link name="l"><gravity>false</gravity><inertial><mass>0.1</mass></inertial>
        <sensor name="cam" type="camera"><always_on>1</always_on><update_rate>5</update_rate>
          <topic>view_cam</topic>
          <camera><horizontal_fov>1.274</horizontal_fov>
            <image><width>1280</width><height>960</height></image>
            <clip><near>1</near><far>500</far></clip></camera>
        </sensor></link></model>
"""


THERMAL_CAMERA = """    <model name="view_camera"><pose>0 0 83 0 1.5707963 1.5707963</pose>
      <link name="l"><gravity>false</gravity><inertial><mass>0.1</mass></inertial>
        <sensor name="cam" type="thermal_camera"><always_on>1</always_on>
          <update_rate>5</update_rate><topic>view_cam</topic>
          <camera><horizontal_fov>0.8727</horizontal_fov>
            <image><width>640</width><height>512</height><format>L16</format></image>
            <clip><near>1</near><far>500</far></clip></camera>
          <plugin filename="gz-sim-thermal-sensor-system" name="gz::sim::systems::ThermalSensor">
            <min_temp>253.15</min_temp><max_temp>673.15</max_temp><resolution>0.01</resolution>
          </plugin>
        </sensor></link></model>
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=Path, required=True)
    ap.add_argument("--poses", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sun-dir", type=float, nargs=3, default=None)
    ap.add_argument("--sun-diffuse", type=float, nargs=3, default=None)
    ap.add_argument("--thermal", action="store_true", help="thermal camera, 16-bit output")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    sdf = args.world.read_text()
    if args.sun_dir:
        sdf = sdf.replace(
            "<direction>-0.4 0.2 -0.9</direction>",
            "<direction>{} {} {}</direction>".format(*args.sun_dir),
        )
    if args.sun_diffuse:
        sdf = sdf.replace(
            "<diffuse>0.9 0.9 0.85 1</diffuse>",
            "<diffuse>{} {} {} 1</diffuse>".format(*args.sun_diffuse),
        )
    camera = THERMAL_CAMERA if args.thermal else CAMERA
    sdf = sdf.replace('    <model name="ground">', camera + '    <model name="ground">', 1)
    world = args.out / "render_world.sdf"
    world.write_text(sdf)
    sim = subprocess.Popen(
        ["gz", "sim", "-s", "-r", "--headless-rendering", str(world)],
        stdout=open(args.out / "gz.log", "w"),
        stderr=subprocess.STDOUT,
    )
    latest: dict = {}

    def on_image(msg: ImgMsg) -> None:
        latest["t"] = time.time()
        latest["msg"] = msg

    node = Node()
    node.subscribe(ImgMsg, "/view_cam", on_image)
    try:
        t0 = time.time()
        while "msg" not in latest:
            if time.time() - t0 > 300:
                raise SystemExit("no camera frame within 300 s")
            time.sleep(0.5)
        poses = json.loads(args.poses.read_text())
        for i, p in enumerate(poses):
            if (args.out / f"{i:04d}.png").exists():
                continue  # rendered by an earlier, interrupted run
            req = Pose()
            req.name = "view_camera"
            req.position.x, req.position.y, req.position.z = p["east"], p["north"], p["height"]
            # heading clockwise from north -> ENU yaw (anticlockwise from east)
            w, x, y, z = nadir_quaternion(math.pi / 2 - p["heading"])
            req.orientation.w, req.orientation.x = w, x
            req.orientation.y, req.orientation.z = y, z
            for _attempt in range(3):  # the service occasionally times out under load
                ok, _ = node.request(f"/world/{WORLD}/set_pose", req, Pose, Boolean, 10000)
                if ok:
                    break
                time.sleep(2)
            if not ok:
                raise SystemExit(f"set_pose failed three times at frame {i}")
            moved = time.time()
            while latest["t"] < moved + 0.5:  # a frame rendered after the move has settled
                time.sleep(0.05)
            m = latest["msg"]
            if args.thermal:  # 16-bit, hundredths of a kelvin
                t = np.frombuffer(m.data, np.uint16).reshape(m.height, m.width)
                Image.fromarray(t).save(args.out / f"{i:04d}.png")
            else:
                a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)[..., :3]
                Image.fromarray(a).save(args.out / f"{i:04d}.png")
        (args.out / "frames.json").write_text(json.dumps(poses, indent=1) + "\n")
        print(f"rendered {len(poses)} frames to {args.out}")
    finally:
        sim.terminate()
        sim.wait(timeout=30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
