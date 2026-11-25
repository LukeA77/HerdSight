"""Drape a real aerial image over the simulated farm's ground. Simulation only.

Usage (inside the simulation image, called by run_flight.sh when GROUND_TEXTURE is set):
  python3 sim/bridge/add_ground_texture.py --world <world.sdf> --texture <image.jpg> --size-m <m>

The generated world's ground is one flat colour (python/herdsight/sim/gazebo.py), so a downward
camera sees nothing it could navigate by. This adds a visual-only square (no collision, so physics
is unchanged) of --size-m metres, centred on the world origin, 1 cm above the ground, textured with
the image (north at the top of the image along +y, the world being ENU). Beyond it the plain ground
remains. The image used is IGN BD ORTHO pasture imagery fetched by scripts/fetch_ground_texture.py
(Etalab Open Licence 2.0, "Source: IGN - BD ORTHO"). The world's fences, trees, rocks and animals
stay where they were; they now stand on real fields, hedges and tracks that do not match them,
which is fine for navigation tests and is not used for counting.

--detail adds a second, smaller square (--detail-m metres, 2 cm above the ground, centred on the
take-off point) with a fine-detail patch made by scripts/make_ground_detail.py: the orthophoto's
layout with static fine texture, standing in for the grass and soil detail a camera sees from
10-20 m up, which 20 cm orthophoto pixels lack. It is a stated assumption of the simulation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

LIFT_M = 0.01  # above the collision plane, so the two never flicker


def add_texture(
    sdf: str, texture_uri: str, size_m: float, name: str = "ground_texture", lift_m: float = LIFT_M
) -> str:
    if f'model name="{name}"' in sdf:
        raise SystemExit(f"world already has {name}")
    block = f"""    <model name="{name}"><static>true</static>
      <pose>0 0 {lift_m} 0 0 0</pose>
      <link name="l"><visual name="v"><cast_shadows>false</cast_shadows>
        <geometry><plane><normal>0 0 1</normal>
          <size>{size_m:.3f} {size_m:.3f}</size></plane></geometry>
        <material><diffuse>1 1 1 1</diffuse><ambient>1 1 1 1</ambient><specular>0 0 0 1</specular>
          <pbr><metal><albedo_map>{texture_uri}</albedo_map>
            <roughness>1</roughness><metalness>0</metalness></metal></pbr>
        </material></visual></link>
    </model>
"""
    marker = '    <model name="ground">'
    if marker not in sdf:
        raise SystemExit("ground model not found in the world")
    return sdf.replace(marker, block + marker, 1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=Path, required=True)
    ap.add_argument("--texture", type=Path, required=True)
    ap.add_argument("--size-m", type=float, required=True)
    ap.add_argument("--detail", type=Path, default=None, help="fine-detail patch near take-off")
    ap.add_argument("--detail-m", type=float, default=120.0)
    ap.add_argument("--fine", type=Path, default=None, help="small fine-grain patch at take-off")
    ap.add_argument("--fine-m", type=float, default=24.0)
    args = ap.parse_args(argv)
    if not args.texture.exists():
        raise SystemExit(f"texture not found: {args.texture}")
    if args.size_m <= 0:
        raise SystemExit("--size-m must be positive")
    uri = args.texture.resolve().as_uri()
    sdf = add_texture(args.world.read_text(), uri, args.size_m)
    print(f"ground texture {args.texture.name}, {args.size_m:.0f} m square, centred on the origin")
    if args.detail is not None:
        if not args.detail.exists():
            raise SystemExit(f"detail patch not found: {args.detail}")
        sdf = add_texture(sdf, args.detail.resolve().as_uri(), args.detail_m, "ground_detail", 0.02)
        print(f"fine-detail patch {args.detail.name}, {args.detail_m:.0f} m square, at take-off")
    if args.fine is not None:
        if not args.fine.exists():
            raise SystemExit(f"fine patch not found: {args.fine}")
        sdf = add_texture(sdf, args.fine.resolve().as_uri(), args.fine_m, "ground_fine", 0.03)
        print(f"fine-grain patch {args.fine.name}, {args.fine_m:.0f} m square, at take-off")
    args.world.write_text(sdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
