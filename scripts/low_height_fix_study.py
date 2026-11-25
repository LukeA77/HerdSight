"""Camera position fix at low heights over the take-off point (development study). Simulation.

Usage:
  python scripts/low_height_fix_study.py poses <run folder>
  (render teach and repeat in the simulation image with sim/bridge/render_views.py over the world
   with the ground texture and detail patches; repeat with another sun)
  python scripts/low_height_fix_study.py evaluate <run folder>

Teach: a climb over the take-off point, frames every 15% of height from 4 m to 83 m, heading 0, as
the drone keeps them before the survey. Repeat: 40 frames at each of 8, 15, 25 and 40 m, offset up
to 30% of the footprint from the take-off point, any heading, rendered under another sun. Each
repeat frame is matched (herdsight.geo.visual_fix.locate) to the teach frames within half to twice
its height. Reports the fixed share and errors per height. Development data for the camera return's
final descent, not a pass/fail test.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from herdsight.geo.visual_fix import Keyframe, locate
from herdsight.utils.config import REPO_ROOT

HFOV, W, H = 1.274, 1280, 960  # render_views camera
OUT_W, OUT_H = 640, 480
HEIGHTS = (8.0, 15.0, 25.0, 40.0)
SEED = 0


def teach_poses() -> list[dict]:
    out, h = [], 4.0
    while h <= 83.0:
        out.append({"east": 0.0, "north": 0.0, "height": h, "heading": 0.0})
        h *= 1.15
    return out


def repeat_poses(max_share: float = 0.3, heights: tuple[float, ...] = HEIGHTS) -> list[dict]:
    rng = np.random.default_rng(SEED)
    out = []
    for h in heights:
        foot = 2 * h * math.tan(HFOV / 2)
        for _ in range(40):
            r, a = rng.uniform(0, max_share * foot), rng.uniform(0, 2 * math.pi)
            out.append(
                {
                    "east": float(r * math.cos(a)),
                    "north": float(r * math.sin(a)),
                    "height": h,
                    "heading": float(rng.uniform(0, 2 * math.pi)),
                }
            )
    return out


def to_map(p: dict) -> np.ndarray:
    s = 2 * p["height"] * math.tan(HFOV / 2) / OUT_W
    hd = p["heading"]
    up = np.array([math.sin(hd), math.cos(hd)])
    right = np.array([math.cos(hd), -math.sin(hd)])
    a = np.column_stack([right * s, -up * s])
    c = np.array([p["east"], p["north"]]) - a @ np.array([OUT_W / 2, OUT_H / 2])
    return np.column_stack([a, c])


def load(folder: Path, i: int) -> np.ndarray:
    im = cv2.imread(str(folder / f"{i:04d}.png"), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(im, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)


def evaluate(run: Path, name: str = "low_height_fix_study") -> int:
    teach = json.loads((run / "teach" / "frames.json").read_text())
    rep = json.loads((run / "repeat" / "frames.json").read_text())
    kfs = [(Keyframe(load(run / "teach", i), to_map(p)), p["height"]) for i, p in enumerate(teach)]
    rng = np.random.default_rng(SEED + 1)
    heights = sorted({p["height"] for p in rep})
    by_h: dict[float, list] = {h: [] for h in heights}
    by_offset: dict[str, list] = {}
    for i, p in enumerate(rep):
        truth = np.array([p["east"], p["north"]])
        near = [k for k, kh in kfs if 0.5 <= kh / p["height"] <= 2.0]
        prior = truth + rng.uniform(-3, 3, 2)
        fix = locate(load(run / "repeat", i), near, prior, max_jump_m=75.0)
        err = None if fix is None else float(np.linalg.norm(fix.east_north - truth))
        by_h[p["height"]].append(err)
        share = np.hypot(*truth) / (2 * p["height"] * math.tan(HFOV / 2))
        lo = min(int(share * 4), 3) * 25
        by_offset.setdefault(f"{int(p['height'])}m_{lo}-{lo + 25}pct", []).append(err)
    note = (
        "Simulation, development study: teach and repeat rendered over the textured world with "
        "detail patches, sun moved; not a test and not a field result."
    )
    out = {"note": note, "by_height": {}}
    for h, errs in by_h.items():
        e = np.array([x for x in errs if x is not None])
        out["by_height"][str(int(h))] = {
            "frames": len(errs),
            "fixed_share": len(e) / len(errs),
            "median_m": float(np.median(e)) if len(e) else None,
            "p90_m": float(np.percentile(e, 90)) if len(e) else None,
        }
    out["fixed_share_by_height_and_offset"] = {
        k: round(sum(x is not None for x in v) / len(v), 2) for k, v in sorted(by_offset.items())
    }
    dest = REPO_ROOT / "results" / "phase5" / f"{name}.json"
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("poses", "evaluate"))
    ap.add_argument("run", type=Path)
    ap.add_argument(
        "--max-share", type=float, default=0.3, help="largest offset, share of footprint"
    )
    ap.add_argument("--heights", type=float, nargs="+", default=list(HEIGHTS))
    ap.add_argument("--name", default="low_height_fix_study")
    args = ap.parse_args(argv)
    if args.step == "poses":
        args.run.mkdir(parents=True, exist_ok=True)
        (args.run / "teach_poses.json").write_text(json.dumps(teach_poses()))
        (args.run / "repeat_poses.json").write_text(
            json.dumps(repeat_poses(args.max_share, tuple(args.heights)))
        )
        print("poses written")
        return 0
    return evaluate(args.run, args.name)


if __name__ == "__main__":
    raise SystemExit(main())
