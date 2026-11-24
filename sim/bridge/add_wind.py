"""Add a steady wind to the simulated farm world and let it push the drone. Simulation only.

Usage (inside the simulation image, called by run_flight.sh when WIND_E or WIND_N is set):
  python3 sim/bridge/add_wind.py --world <world.sdf> --model <x500_base model.sdf> \
      --east <m/s> --north <m/s> [--cda <m^2>]

Gazebo's wind system (gz-sim WindEffects) pushes every link that has wind enabled with
    F = m * k * (wind velocity - link velocity),
a force linear in the relative speed, applied at the centre of mass (gz-sim 8,
src/systems/wind_effects/WindEffects.cc, ApplyWindForce). Real air drag grows with the square of
the relative speed, F = 0.5 * rho * CdA * v^2. So k is set so that the two agree for a drone
holding position in the chosen wind (relative speed = wind speed):
    k = 0.5 * rho * CdA * |wind| / m_body.
Wind is enabled on the drone's body link only (m_body = 2.0 kg in PX4's x500_base), whose mass
the force uses. CdA, the drone's drag area, is an assumption, not a measurement: 0.06 m^2 by
default, which for the 2.06 kg x500 in a 10 m/s wind gives 3.7 N, a lean of about 10 degrees to
hold position. The world is ENU, so --east is +x and --north is +y; the vector is where the air
moves TO (a wind blowing east has --east > 0). The PX4 model copy inside the container is
edited in place; the container is discarded after the flight.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

RHO = 1.225  # kg/m^3, sea-level standard atmosphere
BODY_LINK = "base_link"


def wind_k(speed: float, cda: float, body_mass: float) -> float:
    return 0.5 * RHO * cda * speed / body_mass


def add_to_world(sdf: str, east: float, north: float, k: float) -> str:
    if "<wind>" in sdf or "WindEffects" in sdf:
        raise SystemExit("world already has wind")
    block = (
        f"    <wind><linear_velocity>{east:.3f} {north:.3f} 0</linear_velocity></wind>\n"
        '    <plugin filename="gz-sim-wind-effects-system" name="gz::sim::systems::WindEffects">\n'
        f"      <force_approximation_scaling_factor>{k:.6f}</force_approximation_scaling_factor>\n"
        "      <horizontal><magnitude><time_for_rise>1</time_for_rise>"
        "<sin><amplitude_percent>0</amplitude_percent><period>60</period></sin>"
        '<noise type="gaussian"><mean>0</mean><stddev>0</stddev></noise></magnitude>'
        "<direction><time_for_rise>1</time_for_rise>"
        "<sin><amplitude>0</amplitude><period>60</period></sin>"
        '<noise type="gaussian"><mean>0</mean><stddev>0</stddev></noise></direction>'
        "</horizontal>\n"
        '      <vertical><noise type="gaussian"><mean>0</mean><stddev>0</stddev></noise>'
        "</vertical>\n"
        "    </plugin>\n"
    )
    marker = '<plugin filename="gz-sim-physics-system"'
    if marker not in sdf:
        raise SystemExit("physics plugin not found in the world")
    return sdf.replace(marker, block.lstrip() + "    " + marker, 1)


def enable_on_body(model: str) -> tuple[str, float]:
    m = re.search(rf'<link name="{BODY_LINK}">(.*?)</link>', model, re.S)
    if not m:
        raise SystemExit(f"{BODY_LINK} not found in the model")
    mass = re.search(r"<mass>([0-9.eE+-]+)</mass>", m.group(1))
    if not mass:
        raise SystemExit(f"no mass on {BODY_LINK}")
    if "<enable_wind>" in m.group(1):
        return model, float(mass.group(1))
    start = m.start() + len(f'<link name="{BODY_LINK}">')
    return model[:start] + "<enable_wind>true</enable_wind>" + model[start:], float(mass.group(1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--east", type=float, default=0.0)
    ap.add_argument("--north", type=float, default=0.0)
    ap.add_argument("--cda", type=float, default=0.06, help="drag area Cd*A in m^2 (assumed)")
    args = ap.parse_args(argv)
    speed = math.hypot(args.east, args.north)
    if speed == 0:
        raise SystemExit("no wind given")
    model, body_mass = enable_on_body(args.model.read_text())
    k = wind_k(speed, args.cda, body_mass)
    args.world.write_text(add_to_world(args.world.read_text(), args.east, args.north, k))
    args.model.write_text(model)
    print(
        f"wind {speed:.1f} m/s (east {args.east:+.1f}, north {args.north:+.1f}); k {k:.4f}; "
        f"force while holding position {body_mass * k * speed:.2f} N (CdA {args.cda} m^2, assumed)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
