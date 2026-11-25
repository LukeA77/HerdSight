"""Fix a downward picture camera to the drone's body (Phase 5, camera return home). Simulation only.

Usage (inside the simulation image):
  python3 add_body_camera.py --model <x500_base model.sdf> --plan plan.yaml

The camera of the GPS-loss return (sim/bridge/camera_return.py) must show exactly where the drone
is when each frame is taken. A separate camera model moved to follow the drone lagged by about a
second whenever the simulation slowed, so frames showed where the drone had been. A sensor on the
drone's own body has no such lag; like a real camera without a gimbal it tilts with the drone, and
the return node corrects each frame for the drone's attitude at the frame's time. The camera
points straight down when the drone is level, with the top of the image towards the drone's
front, the same field of view as the survey camera, 660 x 495 pixels, 5 frames per second, topic
body_rgb. The PX4 model copy inside the container is changed, so each run starts from a fresh
container.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml

TOPIC = "body_rgb"
WIDTH_PX, HEIGHT_PX, RATE_HZ = 660, 495, 5
MARKER = '<sensor name="navsat_sensor" type="navsat">'


def add_body_camera(model_sdf: str, fov: float) -> str:
    if model_sdf.count(MARKER) != 1:
        raise ValueError("navsat sensor not found once in the drone model")
    # sensor x axis (optical axis) pitched down onto body -z; sensor z (image up) onto body +x
    sensor = f"""<sensor name="{TOPIC}" type="camera">
        <pose>0 0 -0.05 0 {math.pi / 2:.6f} 0</pose>
        <topic>{TOPIC}</topic>
        <update_rate>{RATE_HZ}</update_rate>
        <always_on>1</always_on>
        <camera>
          <horizontal_fov>{fov:.6f}</horizontal_fov>
          <image><width>{WIDTH_PX}</width><height>{HEIGHT_PX}</height></image>
          <clip><near>0.5</near><far>500</far></clip>
        </camera>
      </sensor>
      """
    return model_sdf.replace(MARKER, sensor + MARKER)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    args = ap.parse_args(argv)
    plan = yaml.safe_load(args.plan.read_text(encoding="utf-8"))
    fov = 2 * math.atan(float(plan["footprint_m"][0]) / 2 / float(plan["height_agl_m"]))
    args.model.write_text(add_body_camera(args.model.read_text(encoding="utf-8"), fov))
    print(f"body camera added: {TOPIC}, horizontal field of view {fov:.3f} rad")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
