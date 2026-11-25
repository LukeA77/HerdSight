"""Camera position fix with frames rendered by the simulator (second feasibility test).

Usage:
  python scripts/gps_return_render_study.py poses <run folder>      # writes teach/repeat poses
  (render both sets in the simulation image with sim/bridge/render_views.py; repeat with a
   different sun)
  python scripts/gps_return_render_study.py evaluate <run folder>   # matches and scores

Same design and pass rule as scripts/gps_return_match_study.py, but the survey keyframes and the
live frames are both rendered by Gazebo's camera over the textured farm world, with the sun moved
for the live frames, so teach and repeat no longer come from one image. Writes
results/phase5/visual_fix_render_study.json. Simulation only.
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

HEIGHT_M, FOOT_W_M, OUT_W, OUT_H = 83.0, 124.0, 640, 480
AREA_M, STEP_E, STEP_N = 400.0, 124.0 * 0.4, 93.0 * 0.3
N_IN, N_OUT, SEED = 200, 50, 0


def teach_poses() -> list[dict]:
    poses = []
    for i, e in enumerate(np.arange(-AREA_M / 2, AREA_M / 2 + 1e-6, STEP_E)):
        lane = np.arange(-AREA_M / 2, AREA_M / 2 + 1e-6, STEP_N)
        for n in lane if i % 2 == 0 else lane[::-1]:
            poses.append(
                {
                    "east": float(e),
                    "north": float(n),
                    "height": HEIGHT_M,
                    "heading": 0.0 if i % 2 == 0 else math.pi,
                }
            )
    return poses


def repeat_poses() -> list[dict]:
    rng = np.random.default_rng(SEED)
    poses = []
    for k in range(N_IN + N_OUT):
        if k < N_IN:
            e, n = rng.uniform(-AREA_M / 2, AREA_M / 2, 2)
        else:
            r, a = rng.uniform(AREA_M / 2 + 100, AREA_M / 2 + 200), rng.uniform(0, 2 * math.pi)
            e, n = r * math.cos(a), r * math.sin(a)
        poses.append(
            {
                "east": float(e),
                "north": float(n),
                "inside": k < N_IN,
                "height": float(HEIGHT_M + rng.uniform(-3, 3)),
                "heading": float(rng.uniform(0, 2 * math.pi)),
            }
        )
    return poses


def to_map(p: dict) -> np.ndarray:
    """Frame pixel (640 x 480) -> metres east and north, from the camera pose."""
    s = FOOT_W_M * p["height"] / HEIGHT_M / OUT_W  # metres per pixel at this height
    h = p["heading"]
    up = np.array([math.sin(h), math.cos(h)])
    right = np.array([math.cos(h), -math.sin(h)])
    a = np.column_stack([right * s, -up * s])  # d(x) along right, d(y) down the image
    c = np.array([p["east"], p["north"]]) - a @ np.array([OUT_W / 2, OUT_H / 2])
    return np.column_stack([a, c])


def load(folder: Path, i: int) -> np.ndarray:
    im = cv2.imread(str(folder / f"{i:04d}.png"), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(im, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)


def moved_animals(f: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    a = f.astype(np.float32)
    sig = rng.uniform(0, 1.5)
    if sig > 0.1:
        a = cv2.GaussianBlur(a, (0, 0), sig)
    px_per_m = OUT_W / FOOT_W_M
    for _ in range(int(rng.integers(0, 16))):
        c = tuple(int(v) for v in rng.uniform([0, 0], [OUT_W, OUT_H]))
        ln = rng.uniform(1, 2) * px_per_m
        cv2.ellipse(
            a,
            c,
            (int(ln / 2), int(ln / 4)),
            float(rng.uniform(0, 180)),
            0,
            360,
            float(rng.uniform(10, 50)),
            -1,
        )
    return np.clip(a, 0, 255).astype(np.uint8)


def evaluate(
    run: Path,
    max_jump_m: float | None = None,
    name: str = "visual_fix_render_study",
    note: str = "",
) -> int:
    teach = json.loads((run / "teach" / "frames.json").read_text())
    rep = json.loads((run / "repeat" / "frames.json").read_text())
    kfs = [Keyframe(load(run / "teach", i), to_map(p)) for i, p in enumerate(teach)]
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for i, p in enumerate(rep):
        truth = np.array([p["east"], p["north"]])
        ang = rng.uniform(0, 2 * math.pi)
        prior = truth + rng.uniform(0, 50) * np.array([math.cos(ang), math.sin(ang)])
        fix = locate(moved_animals(load(run / "repeat", i), rng), kfs, prior, max_jump_m=max_jump_m)
        err = None if fix is None else float(np.linalg.norm(fix.east_north - truth))
        rows.append(
            {
                "inside": p["inside"],
                "fixed": fix is not None,
                "error_m": err,
                "inliers": None if fix is None else fix.inliers,
            }
        )
    ins = [r for r in rows if r["inside"]]
    errs = np.array([r["error_m"] for r in ins if r["fixed"]])
    wrong = sum(1 for r in rows if r["fixed"] and r["error_m"] > 20)
    out = {
        "note": "Simulation: keyframes and live frames rendered by Gazebo over real pasture "
        "imagery, sun moved between them; not a flight result." + note,
        "max_jump_m": max_jump_m,
        "keyframes": len(kfs),
        "inside_frames": len(ins),
        "inside_fixed_share": float(np.mean([r["fixed"] for r in ins])),
        "inside_error_median_m": float(np.median(errs)) if len(errs) else None,
        "inside_error_p95_m": float(np.percentile(errs, 95)) if len(errs) else None,
        "outside_frames": len(rows) - len(ins),
        "outside_fixed_share": float(np.mean([r["fixed"] for r in rows if not r["inside"]])),
        "wrong_fix_share_all": wrong / len(rows),
    }
    out["passes"] = bool(
        out["inside_fixed_share"] >= 0.9
        and out["inside_error_median_m"] is not None
        and out["inside_error_median_m"] <= 3
        and out["inside_error_p95_m"] <= 10
        and out["wrong_fix_share_all"] <= 0.01
    )
    out["rows"] = rows
    dest = REPO_ROOT / "results" / "phase5" / f"{name}.json"
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("poses", "evaluate"))
    ap.add_argument("run", type=Path)
    ap.add_argument("--max-jump", type=float, default=None, help="refuse fixes further (m)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the repeat poses and noise")
    ap.add_argument("--name", default="visual_fix_render_study", help="results/phase5/<name>.json")
    ap.add_argument("--note", default="", help="text appended to the result's note")
    args = ap.parse_args(argv)
    global SEED
    SEED = args.seed
    if args.step == "poses":
        args.run.mkdir(parents=True, exist_ok=True)
        (args.run / "teach_poses.json").write_text(json.dumps(teach_poses()))
        (args.run / "repeat_poses.json").write_text(json.dumps(repeat_poses()))
        print("poses written")
        return 0
    return evaluate(args.run, args.max_jump, args.name, args.note)


if __name__ == "__main__":
    raise SystemExit(main())
