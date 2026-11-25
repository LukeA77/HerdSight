"""Give the simulated farm night-time temperatures for a thermal camera. Simulation only.

Usage (inside the simulation image):
  python3 add_thermal.py --world <world.sdf> [--air-k 281] [--seed 0]

Gazebo's thermal camera shows each visual at the temperature set by the Thermal system on it, and
everything else at the world's air temperature. This sets: the air (and so the ground) at --air-k
(281 K, about 8 degrees C, a cool night in spring or autumn); each animal's body and head at its own
coat surface temperature, drawn between 301 and 306 K (28-33 degrees C: the coat surface of cattle
at night is well below the 38.5 degree core temperature, as infrared surveys of cattle show); trees
4 K and rocks and objects 6 K above the air (they keep some of the day's heat); fence rails at the
air temperature. These are stated assumptions of the simulation, not measurements.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

COAT_K = (301.0, 306.0)
TREE_ABOVE_AIR_K = 4.0
OBJECT_ABOVE_AIR_K = 6.0


def thermal(temperature_k: float) -> str:
    return (
        '<plugin filename="gz-sim-thermal-system" name="gz::sim::systems::Thermal">'
        f"<temperature>{temperature_k:.2f}</temperature></plugin>"
    )


def add_thermal(
    sdf: str, air_k: float, seed: int = 0, object_above_k: float = OBJECT_ABOVE_AIR_K
) -> str:
    rng = np.random.default_rng(seed)
    out = []
    pos = 0
    pattern = re.compile(r'<model name="(animal_\d+|tree_\d+|object_\d+)">')
    for m in pattern.finditer(sdf):
        start = m.start()
        end = sdf.index("</model>", start)
        out.append(sdf[pos:start])
        block = sdf[start:end]
        kind = m.group(1).split("_")[0]
        if kind == "animal":
            temp = float(rng.uniform(*COAT_K))
        elif kind == "tree":
            temp = air_k + TREE_ABOVE_AIR_K
        else:
            temp = air_k + object_above_k
        block = block.replace("</visual>", thermal(temp) + "</visual>")
        out.append(block)
        pos = end
    out.append(sdf[pos:])
    sdf = "".join(out)
    marker = "  <world name="
    i = sdf.index(marker)
    j = sdf.index(">", i) + 1
    atmosphere = (
        f'\n    <atmosphere type="adiabatic"><temperature>{air_k:.2f}</temperature></atmosphere>'
    )
    return sdf[:j] + atmosphere + sdf[j:]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=Path, required=True)
    ap.add_argument("--air-k", type=float, default=281.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--object-above-k", type=float, default=OBJECT_ABOVE_AIR_K, help="rocks and objects"
    )
    args = ap.parse_args(argv)
    sdf = add_thermal(args.world.read_text(), args.air_k, args.seed, args.object_above_k)
    args.world.write_text(sdf)
    print(f"thermal: air {args.air_k:.1f} K, animals {COAT_K[0]:.0f}-{COAT_K[1]:.0f} K")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
