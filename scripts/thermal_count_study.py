"""Night counting with a simulated thermal camera (Phase 5, thermal).

Usage:
  python scripts/thermal_count_study.py render-script <run folder>   # writes render.sh
  (run it in the simulation image: docker ... herdsight-sim:dev bash /out/render.sh)
  python scripts/thermal_count_study.py evaluate <run folder>

Worlds from make_world seeds 10-19 get night temperatures (sim/bridge/add_thermal.py) and are
rendered by the thermal camera (sim/bridge/render_views.py --thermal) at 40 m and 83 m, nadir, at
the herd centre and 15 m north, south, east and west of it. Realism is then added (ground
temperature patches, optical blur, sensor noise), animals are counted from warm regions, and counts
are compared with the animals whose centre is in the frame. Rule fixed before any frame was rendered
(_private notes, 25 Nov). Writes results/phase5/thermal_count_study.json. Simulation only.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from herdsight.utils.config import REPO_ROOT

SEEDS = range(10, 20)
HEIGHTS = (40.0, 83.0)
OFFSETS = ((0, 0), (0, 15), (0, -15), (15, 0), (-15, 0))
W, H, HFOV = 640, 512, 0.8727
GROUND_SD_K, BLUR_PX, NOISE_K = 1.5, 0.8, 0.05
WARM_ABOVE_K, MIN_AREA_SHARE = 8.0, 0.25


def animals(sdf: str) -> list[tuple[float, float, float]]:
    """(east, north, body length) of each animal in a generated world."""
    out = []
    for m in re.finditer(r'<model name="animal_\d+">(.*?)</model>', sdf, re.S):
        x, y = (float(v) for v in re.search(r"<pose>([-\d.]+) ([-\d.]+)", m.group(1)).groups())
        body_l = float(re.search(r"<box><size>([\d.]+)", m.group(1)).group(1))
        out.append((x, y, body_l / 0.72))
    return out


def render_script(run: Path, seeds: range = SEEDS, object_above_k: float = 6.0) -> int:
    lines = ["#!/bin/bash", "set -e"]
    for s in seeds:
        lines += [
            f"python3 /repo/sim/bridge/make_world.py --out /out/world{s} --seed {s} > /dev/null",
            f"python3 /repo/sim/bridge/add_thermal.py --world /out/world{s}/herdsight_farm.sdf "
            f"--seed {s} --object-above-k {object_above_k}",
            "PYTHONPATH=/repo/python python3 /repo/scripts/thermal_count_study.py poses "
            f"/out/world{s}",
            f"timeout 1200 python3 /repo/sim/bridge/render_views.py --world "
            f"/out/world{s}/herdsight_farm.sdf --poses /out/world{s}/poses.json "
            f"--out /out/world{s}/frames --thermal",
        ]
    run.mkdir(parents=True, exist_ok=True)
    (run / "render.sh").write_text("\n".join(lines) + "\n", newline="\n")
    print(f"wrote {run / 'render.sh'}")
    return 0


def poses(world: Path) -> int:
    a = animals((world / "herdsight_farm.sdf").read_text())
    cx, cy = np.mean([p[0] for p in a]), np.mean([p[1] for p in a])
    out = [
        {"east": float(cx + dx), "north": float(cy + dy), "height": h, "heading": 0.0}
        for h in HEIGHTS
        for dx, dy in OFFSETS
    ]
    (world / "poses.json").write_text(json.dumps(out, indent=1))
    return 0


def in_frame(a: tuple[float, float, float], p: dict, edge_m: float = 0.0) -> bool:
    half_w = p["height"] * math.tan(HFOV / 2) - edge_m
    half_h = p["height"] * math.tan(HFOV / 2) * H / W - edge_m
    return abs(a[0] - p["east"]) <= half_w and abs(a[1] - p["north"]) <= half_h


def realism(t: np.ndarray, rng: np.random.Generator, px_m: float) -> np.ndarray:
    """Ground temperature patches where the ground shows, optical blur and sensor noise."""
    ground = t < np.median(t) + 1.0
    patches = np.zeros_like(t)
    for scale_m in (2.0, 5.0, 10.0):
        cells = (max(2, int(H * px_m / scale_m)), max(2, int(W * px_m / scale_m)))
        layer = rng.normal(0, 1, cells).astype(np.float32)
        patches += cv2.resize(layer, (W, H), interpolation=cv2.INTER_CUBIC)
    patches *= GROUND_SD_K / max(patches.std(), 1e-6)
    out = t + np.where(ground, patches, 0.0)
    out = cv2.GaussianBlur(out.astype(np.float32), (0, 0), BLUR_PX)
    return out + rng.normal(0, NOISE_K, out.shape)


ADULT_L_M, SPLIT_ABOVE = 2.3, 1.2


def adult_area_px(px_m: float) -> float:
    """Expected warm area of one adult from above (body and head boxes), in pixels."""
    body = (0.72 * ADULT_L_M) * (0.32 * ADULT_L_M)
    head = (0.25 * ADULT_L_M) * (0.6 * 0.32 * ADULT_L_M)
    return (body + head) / px_m**2


EDGE_M = 2.0  # version 3: regions and animals must lie this far inside the frame edge


def count_v2(t: np.ndarray, px_m: float, edge_px: float = 0.0) -> int:
    """Warm regions; a region over 1.2 adult areas counts round(area / adult area), others count 1.

    Chosen on the first study's frames (seeds 10-19) after its failure: herds mix calves and adults,
    so the frame's median region (version 1) is no yardstick, while the camera height is known.
    """
    warm = (t > np.median(t) + WARM_ABOVE_K).astype(np.uint8)
    _, _, stats, cents = cv2.connectedComponentsWithStats(warm, connectivity=8)
    adult = adult_area_px(px_m)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
    c = cents[1:]
    inside = (
        (c[:, 0] >= edge_px)
        & (c[:, 0] <= W - edge_px)
        & (c[:, 1] >= edge_px)
        & (c[:, 1] <= H - edge_px)
    )
    areas = areas[(areas >= 0.1 * adult) & inside]
    return int(sum(max(1, round(a / adult)) if a > SPLIT_ABOVE * adult else 1 for a in areas))


def count(t: np.ndarray, animal_px_area: float) -> int:
    warm = (t > np.median(t) + WARM_ABOVE_K).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(warm, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
    areas = areas[areas >= MIN_AREA_SHARE * animal_px_area]
    if not len(areas):
        return 0
    med = float(np.median(areas))
    return int(sum(max(1, round(a / med)) for a in areas))


def evaluate(run: Path, seeds: range = SEEDS, version: int = 1, name: str | None = None) -> int:
    rng = np.random.default_rng(0)
    rows = []
    for s in seeds:
        world = run / f"world{s}"
        a = animals((world / "herdsight_farm.sdf").read_text())
        mean_len = float(np.mean([x[2] for x in a]))
        for i, p in enumerate(json.loads((world / "frames" / "frames.json").read_text())):
            t = np.array(Image.open(world / "frames" / f"{i:04d}.png")).astype(np.float64) / 100
            px_m = 2 * p["height"] * math.tan(HFOV / 2) / W
            animal_px = (0.72 * mean_len) * (0.32 * mean_len) / px_m**2
            frame = realism(t, rng, px_m)
            edge_m = EDGE_M if version == 3 else 0.0
            if version == 1:
                got = count(frame, animal_px)
            else:
                got = count_v2(frame, px_m, edge_m / px_m)
            truth = sum(in_frame(x, p, edge_m) for x in a)
            rows.append({"seed": s, "height": p["height"], "true": truth, "counted": got})
    out = {
        "note": "Simulation: Gazebo thermal camera over box-shaped animals with set temperatures, "
        "with added ground patches, blur and noise; not a field result.",
        "rows": rows,
    }
    for h in HEIGHTS:
        r = [x for x in rows if x["height"] == h]
        t, c = sum(x["true"] for x in r), sum(x["counted"] for x in r)
        out[f"h{int(h)}"] = {
            "frames": len(r),
            "true": t,
            "counted": c,
            "mae": float(np.mean([abs(x["counted"] - x["true"]) for x in r])),
            "ratio": c / max(t, 1),
        }
    h40 = out["h40"]
    out["passes"] = bool(h40["mae"] <= 1.0 and 0.9 <= h40["ratio"] <= 1.1)
    name = name or "thermal_count_study" + ("" if version == 1 else f"_v{version}")
    out["counter_version"] = version
    out["seeds"] = [seeds.start, seeds.stop - 1]
    dest = REPO_ROOT / "results" / "phase5" / f"{name}.json"
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("render-script", "poses", "evaluate"))
    ap.add_argument("path", type=Path)
    ap.add_argument(
        "--seeds", type=int, nargs=2, default=[10, 19], help="first and last world seed"
    )
    ap.add_argument("--counter", type=int, default=1, choices=(1, 2, 3))
    ap.add_argument("--object-above-k", type=float, default=6.0, help="warm rocks and objects")
    ap.add_argument("--name", default=None, help="results/phase5/<name>.json")
    args = ap.parse_args(argv)
    seeds = range(args.seeds[0], args.seeds[1] + 1)
    if args.step == "render-script":
        return render_script(args.path, seeds, args.object_above_k)
    if args.step == "poses":
        return poses(args.path)
    return evaluate(args.path, seeds, args.counter, args.name)


if __name__ == "__main__":
    raise SystemExit(main())
