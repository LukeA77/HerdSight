"""Which camera fixes are wrong, and can the known height and heading tell? (development study)

Usage: python scripts/camera_fix_consistency_study.py <run folder> [<run folder> ...]

Uses recorded simulated flights (sim/bridge/camera_return.py --record). Keyframes are rebuilt from
the frames taken while GPS was good, exactly as in scripts/camera_return_offline.py. Every frame
after GPS loss above 2 m is matched twice: once with the prior at the true position, and once with
the prior pushed 20 m off in a random direction (as when the chained position has drifted). For each
fix it records the error against the true position and four checks the node could make without
knowing the truth:
  - inliers, and the inlier share (RANSAC inliers / ratio-test matches);
  - scale error: the fix's metres per pixel against the one expected from the height (per cent);
  - rotation error: the fix's image heading against the one expected from the attitude (degrees).
Writes results/phase5/camera_fix_consistency_study.json with the distributions for good fixes
(error <= 3 m) and wrong fixes (error > 8 m), and how many of each a set of limits keeps.
Simulation, development data only: limits chosen here are tested on fresh flights afterwards.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

import camera_return_offline as cro  # noqa: E402  (same folder)
from herdsight.geo.visual_fix import locate
from herdsight.utils.config import REPO_ROOT

HEIGHT_RATIO = 2.0
GOOD_M, WRONG_M = 3.0, 8.0
MIN_HEIGHT_M = 2.0  # frames below this are left out (the drone is landing or on the ground)


def scale_rot(a: np.ndarray) -> tuple[float, float]:
    """Metres per pixel and heading (rad) of a 2x2 similarity part."""
    return math.sqrt(abs(np.linalg.det(a))), math.atan2(a[1, 0], a[0, 0])


def study(run: Path, rng: np.random.Generator) -> list[dict]:
    rows = [json.loads(x) for x in open(run / "frames" / "frames.jsonl")]
    states = sorted(rows, key=lambda r: r["state_us"])
    st_t = np.array([r["state_us"] for r in states])

    def state_at(t: float) -> dict:
        return states[int(np.clip(np.searchsorted(st_t, t), 0, len(states) - 1))]

    good = [r for r in rows if r["gps_ok"] and r["true_en"]]
    off = np.median([np.array(r["true_en"]) - [r["ned"][1], r["ned"][0]] for r in good], axis=0)

    def image(r: dict) -> np.ndarray:
        return np.array(cro.Image.open(run / "frames" / f"{r['image_us']}.png"))

    kfs, info = [], []
    for r in rows:
        if not r["gps_ok"]:
            break
        s = state_at(r["image_us"])
        o, th, sl, h = cro.geometry(np.array(s["ned"]), np.array(s["q"]))
        if h < cro.MIN_KEYFRAME_HEIGHT_M:
            continue
        centre = np.array([s["ned"][1], s["ned"][0]]) + o
        need = cro.NEW_KEYFRAME_SHARE * 2 * sl * math.tan(cro.FOV / 2)
        if any(
            abs(kh - h) <= cro.SAME_HEIGHT_SHARE * h
            and math.hypot(ke - centre[0], kn - centre[1]) < need
            for ke, kn, kh in info
        ):
            continue
        kfs.append(cro.Keyframe(image(r), cro.frame_to_map(centre[0], centre[1], sl, th)))
        info.append((centre[0], centre[1], h))
    out = []
    for r in rows:
        if r["gps_ok"] or not r["true_en"]:
            continue
        s = state_at(r["image_us"])
        o, th, sl, h = cro.geometry(np.array(s["ned"]), np.array(s["q"]))
        if h < MIN_HEIGHT_M:
            continue
        truth = np.array(r["true_en"]) - off
        expect = cro.frame_to_map(0.0, 0.0, sl, th)[:, :2]
        e_scale, e_rot = scale_rot(expect)
        near = [
            k
            for k, (_, _, kh) in zip(kfs, info, strict=True)
            if 1 / HEIGHT_RATIO <= kh / max(h, 1) <= HEIGHT_RATIO
        ]
        live = image(r)
        for prior_kind in ("true", "off20"):
            p = truth.copy()
            if prior_kind == "off20":
                a = rng.uniform(0, 2 * math.pi)
                p += 20 * np.array([math.cos(a), math.sin(a)])
            fix = locate(live, near, p + o, max_jump_m=75.0)
            if fix is None:
                continue
            f_scale, f_rot = scale_rot(fix.to_map[:, :2])
            rot = math.degrees((f_rot - e_rot + math.pi) % (2 * math.pi) - math.pi)
            out.append(
                {
                    "run": run.name, "prior": prior_kind, "height": round(h, 1),
                    "error_m": float(np.linalg.norm(fix.east_north - o - truth)),
                    "inliers": fix.inliers, "inlier_share": fix.inliers / max(fix.matches, 1),
                    "scale_err_pct": 100 * abs(f_scale / e_scale - 1), "rot_err_deg": abs(rot),
                }
            )  # fmt: skip
    return out


def summary(rows: list[dict]) -> dict:
    good = [r for r in rows if r["error_m"] <= GOOD_M]
    wrong = [r for r in rows if r["error_m"] > WRONG_M]
    keys = ("inliers", "inlier_share", "scale_err_pct", "rot_err_deg")

    def q(rs: list[dict], k: str) -> list[float]:
        return (
            [round(float(np.percentile([r[k] for r in rs], p)), 3) for p in (10, 50, 90)]
            if rs
            else []
        )

    out = {
        "fixes": len(rows), "good": len(good), "wrong": len(wrong),
        "good_p10_p50_p90": {k: q(good, k) for k in keys},
        "wrong_p10_p50_p90": {k: q(wrong, k) for k in keys},
        "limits": [],
    }  # fmt: skip
    grid = itertools.product((10, 25, 1000), (10, 20, 1000), (40, 100, 150, 250), (0.0, 0.5, 0.6))
    for sc, rt, inl, sh in grid:
        keep = [
            r["scale_err_pct"] <= sc
            and r["rot_err_deg"] <= rt
            and r["inliers"] >= inl
            and r["inlier_share"] >= sh
            for r in good + wrong
        ]
        n_good = sum(keep[: len(good)])
        n_wrong = sum(keep[len(good) :])
        out["limits"].append(
            {
                "scale_pct": sc, "rot_deg": rt, "min_inliers": inl, "min_share": sh,
                "good_kept": round(n_good / max(len(good), 1), 3),
                "wrong_kept": round(n_wrong / max(len(wrong), 1), 3),
            }
        )  # fmt: skip
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--name", default="camera_fix_consistency_study")
    ap.add_argument("--rows", type=Path, default=None, help="also save every fix to this file")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(0)
    rows: list[dict] = []
    for run in args.runs:
        got = study(run, rng)
        print(run.name, len(got), "fixes", flush=True)
        rows += got
    res = {
        "note": "Simulation, development study on recorded flights; not a test and not a field "
        "result. Good fix: error <= 3 m; wrong fix: error > 8 m.",
        "runs": [r.name for r in args.runs],
        "all": summary(rows),
        "true_prior": summary([r for r in rows if r["prior"] == "true"]),
        "prior_20m_off": summary([r for r in rows if r["prior"] == "off20"]),
    }
    dest = REPO_ROOT / "results" / "phase5" / f"{args.name}.json"
    dest.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    if args.rows:
        args.rows.write_text(json.dumps(rows), encoding="utf-8")
    a = res["all"]
    print(json.dumps({k: a[k] for k in ("fixes", "good", "wrong", "good_p10_p50_p90",
                                        "wrong_p10_p50_p90")}, indent=1))  # fmt: skip
    best = sorted((x for x in a["limits"] if x["good_kept"] >= 0.7), key=lambda x: x["wrong_kept"])
    print("keeping >= 70% of good fixes, fewest wrong:", best[:8])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
