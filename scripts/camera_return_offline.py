"""Offline study of the camera return's position fixes on a recorded simulated flight.

Usage: python scripts/camera_return_offline.py <run folder> [--lag-s 0 0.25 0.5] [--nearest 6 12]

Reads <run>/frames (written by sim/bridge/camera_return.py --record: every frame with PX4's state
and Gazebo's true position), rebuilds the keyframes exactly as the node does from the frames taken
while GPS was good, and tries a fix on every frame after GPS was lost, for each combination of
settings. The same recording is used for every setting, so settings are compared on identical
frames (single closed-loop flights vary a lot from run to run). Errors are against Gazebo's true
position, shifted into PX4's frame by the median difference while GPS was good. Development data
only: a setting chosen here is then tested on fresh flights with a rule written beforehand.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from herdsight.geo.visual_fix import Keyframe, locate  # noqa: E402

WIDTH_PX, HEIGHT_PX = 660, 495
FOV = 1.250  # horizontal field of view of the body camera (rad), from add_body_camera
NEW_KEYFRAME_SHARE, SAME_HEIGHT_SHARE, MIN_KEYFRAME_HEIGHT_M = 0.4, 0.15, 4.0


def rotation(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def geometry(ned: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Image-centre ground offset (east, north), image-top ENU yaw, slant distance, height."""
    r = rotation(q)
    axis = r @ np.array([0.0, 0.0, 1.0])
    h = -ned[2]
    c = max(axis[2], 0.2)
    fwd = r @ np.array([1.0, 0.0, 0.0])
    return h * np.array([axis[1], axis[0]]) / c, math.atan2(fwd[0], fwd[1]), h / c, h


def frame_to_map(east: float, north: float, slant: float, theta: float) -> np.ndarray:
    s = 2 * slant * math.tan(FOV / 2) / WIDTH_PX
    up = np.array([math.cos(theta), math.sin(theta)])
    right = np.array([math.sin(theta), -math.cos(theta)])
    a = np.column_stack([right * s, -up * s])
    c = np.array([east, north]) - a @ np.array([WIDTH_PX / 2, HEIGHT_PX / 2])
    return np.column_stack([a, c])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--lag-s", type=float, nargs="+", default=[0.0])
    ap.add_argument("--nearest", type=int, nargs="+", default=[6])
    ap.add_argument("--height-ratio", type=float, nargs="+", default=[1.5])
    args = ap.parse_args(argv)
    rows = [json.loads(x) for x in open(args.run / "frames" / "frames.jsonl")]
    states = sorted(rows, key=lambda r: r["state_us"])
    st_t = np.array([r["state_us"] for r in states])

    def state_at(t: float) -> dict:
        i = int(np.clip(np.searchsorted(st_t, t), 0, len(states) - 1))
        return states[i]

    good = [r for r in rows if r["gps_ok"] and r["true_en"]]
    off = np.median([np.array(r["true_en"]) - [r["ned"][1], r["ned"][0]] for r in good], axis=0)
    img = {}

    def image(r: dict) -> np.ndarray:
        if r["image_us"] not in img:
            img[r["image_us"]] = np.array(Image.open(args.run / "frames" / f"{r['image_us']}.png"))
        return img[r["image_us"]]

    results = []
    for lag, nearest, hr in itertools.product(args.lag_s, args.nearest, args.height_ratio):
        kfs, info = [], []
        for r in rows:
            if not r["gps_ok"]:
                break
            s = state_at(r["image_us"] - lag * 1e6)
            o, th, sl, h = geometry(np.array(s["ned"]), np.array(s["q"]))
            if h < MIN_KEYFRAME_HEIGHT_M:
                continue
            en = np.array([s["ned"][1], s["ned"][0]])
            centre = en + o
            need = NEW_KEYFRAME_SHARE * 2 * sl * math.tan(FOV / 2)
            if any(
                abs(kh - h) <= SAME_HEIGHT_SHARE * h
                and math.hypot(ke - centre[0], kn - centre[1]) < need
                for ke, kn, kh in info
            ):
                continue
            kfs.append(Keyframe(image(r), frame_to_map(centre[0], centre[1], sl, th)))
            info.append((centre[0], centre[1], h))
        errs, tried, by_h = [], 0, {}
        prior = None
        for r in rows:
            if r["gps_ok"] or not r["true_en"]:
                continue
            s = state_at(r["image_us"] - lag * 1e6)
            o, th, sl, h = geometry(np.array(s["ned"]), np.array(s["q"]))
            if h < 5:
                continue
            tried += 1
            en = np.array([s["ned"][1], s["ned"][0]])
            p = prior if prior is not None else en
            sim = [
                k for k, (_, _, kh) in zip(kfs, info, strict=True) if 1 / hr <= kh / max(h, 1) <= hr
            ]
            fix = locate(image(r), sim, p + o, nearest=nearest, max_jump_m=75.0)
            band = int(h // 20) * 20
            by_h.setdefault(band, [0, 0])
            by_h[band][1] += 1
            if fix is None:
                continue
            pos = fix.east_north - o
            prior = pos
            truth = np.array(r["true_en"]) - off
            errs.append(float(np.linalg.norm(pos - truth)))
            by_h[band][0] += 1
        e = np.array(errs)
        results.append(
            {
                "lag_s": lag, "nearest": nearest, "height_ratio": hr, "keyframes": len(kfs),
                "frames": tried, "fixed_share": len(e) / max(tried, 1),
                "median_m": float(np.median(e)) if len(e) else None,
                "p90_m": float(np.percentile(e, 90)) if len(e) else None,
                "fixed_by_height": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_h.items())},
            }
        )  # fmt: skip
        print(json.dumps(results[-1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
