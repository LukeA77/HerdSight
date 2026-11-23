"""Add the stabilised bounding-box camera to a Gazebo world (Phase 5, step 8).

Usage (inside the simulation image): python3 add_survey_camera.py --plan plan.yaml --world w.sdf

The camera is a separate model with no collision or gravity, so the drone flies as it does without
it (a static model's sensors did not follow its pose).
sim/bridge/camera_bridge.py keeps it at the drone's position, pointing straight down with the top
of the image along the survey heading: an ideal gimbal. PX4's simulated gimbal did not move when
commanded, and a camera fixed to the body looks about 10 degrees off vertical at survey speed.

Two sensors report the visible part of each labelled animal and its whole box, so the visible
share can be computed. The field of view comes from the plan's footprint and height. The image has
fewer pixels than the design camera and the same aspect ratio to within 0.2%. Simulation only.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import yaml

WIDTH_PX, HEIGHT_PX = 660, 495
RATE_HZ = 5
MODEL_NAME = "survey_camera"


def nadir_quaternion(heading_rad: float) -> tuple[float, float, float, float]:
    """(w, x, y, z): camera x axis straight down, image top along the heading."""
    ch, sh = math.cos(heading_rad / 2), math.sin(heading_rad / 2)
    cp = sp = math.sqrt(0.5)
    return ch * cp, -sh * sp, ch * sp, sh * cp


def sensor(name: str, box_type: str, fov: float) -> str:
    return f"""<sensor name="{name}" type="boundingbox_camera">
          <topic>{name}</topic>
          <update_rate>{RATE_HZ}</update_rate>
          <always_on>1</always_on>
          <camera>
            <box_type>{box_type}</box_type>
            <horizontal_fov>{fov:.6f}</horizontal_fov>
            <image><width>{WIDTH_PX}</width><height>{HEIGHT_PX}</height></image>
            <clip><near>1</near><far>500</far></clip>
          </camera>
        </sensor>
"""


def add_camera(sdf: str, across_m: float, height_m: float, heading_rad: float) -> str:
    fov = 2 * math.atan(across_m / 2 / height_m)
    marker = "</world>"
    if sdf.count(marker) != 1:
        raise ValueError("world end not found")
    sensors = sensor("survey_bbox_visible", "2d", fov) + sensor("survey_bbox_full", "full_2d", fov)
    model = f"""<model name="{MODEL_NAME}">
      <pose>0 0 {height_m:.3f} 0 {math.pi / 2:.6f} {heading_rad:.6f}</pose>
      <link name="link">
        <gravity>false</gravity>
        <inertial><mass>1</mass></inertial>
        {sensors}      </link>
    </model>
  """
    return sdf.replace(marker, model + marker)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--world", type=Path, required=True)
    args = ap.parse_args(argv)
    plan = yaml.safe_load(args.plan.read_text(encoding="utf-8"))
    text = add_camera(
        args.world.read_text(encoding="utf-8"),
        float(plan["footprint_m"][0]),
        float(plan["height_agl_m"]),
        float(plan["heading_enu_rad"]),
    )
    args.world.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
