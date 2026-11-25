"""Can the downward camera fix its position at survey height by matching to the survey's keyframes?

Usage: python scripts/gps_return_match_study.py   (after scripts/fetch_ground_texture.py)

Feasibility study for the GPS-loss return home (_drafts/gps-loss-camera-return-spec.md), with the
rule fixed before the run. Map: the IGN BD ORTHO pasture image (0.206 m per pixel, Etalab Open
Licence 2.0). Teach: a lawnmower survey of the central 400 x 400 m at 83 m (footprint 124 x 93 m,
70% forward and 60% side overlap), each keyframe stored greyscale at 640 x 480. Repeat: 400 live
frames inside the taught area at a random heading, 83 +/- 3 m, brightness and contrast x0.8-1.2,
blur 0-1.5 px and 0-15 dark ellipses of 1-2 m (moved animals), matched against the 6 keyframes
nearest a prior that is off by up to 50 m; plus 100 frames 100-200 m outside the taught area,
which should get no fix. Pass: >= 90% fixed inside, median error <= 3 m, 95th percentile <= 10 m,
wrong fixes (> 20 m off) <= 1% of all frames. Teach and repeat come from the same image, so this is
necessary, not sufficient. Writes results/phase5/visual_fix_study.json. Simulation of imagery only.
"""

from __future__ import annotations

import json
import math

import cv2
import numpy as np

from herdsight.geo.visual_fix import Keyframe, locate
from herdsight.utils.config import REPO_ROOT
from herdsight.utils.paths import storage_path

HEIGHT_M, FOOT_W_M, FOOT_H_M, OUT_W, OUT_H = 83.0, 124.0, 93.0, 640, 480
AREA_M, FWD_OVERLAP, SIDE_OVERLAP = 400.0, 0.7, 0.6
N_IN, N_OUT, SEED = 400, 100, 0


def frame(
    img: np.ndarray,
    m_per_px: float,
    east: float,
    north: float,
    heading: float,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """640x480 view centred at (east, north) metres from the image centre, image top along
    `heading` (radians clockwise from north); returns the frame and its pixel -> metres affine."""
    hpx, wpx = img.shape[:2]
    cx, cy = wpx / 2 + east / m_per_px, hpx / 2 - north / m_per_px
    s = OUT_W / (FOOT_W_M / m_per_px) * scale
    rot = cv2.getRotationMatrix2D((cx, cy), math.degrees(heading), s)
    rot[:, 2] += [OUT_W / 2 - cx, OUT_H / 2 - cy]
    out = cv2.warpAffine(img, rot, (OUT_W, OUT_H), borderMode=cv2.BORDER_REFLECT)
    to_px = cv2.invertAffineTransform(rot)  # frame pixel -> image pixel
    # image pixel (x, y) -> metres: east = (x - w/2) m, north = (h/2 - y) m
    to_m = np.array([[m_per_px, 0, -wpx / 2 * m_per_px], [0, -m_per_px, hpx / 2 * m_per_px]])
    return out, to_m @ np.vstack([to_px, [0, 0, 1]])


def perturb(f: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    a = f.astype(np.float32) * rng.uniform(0.8, 1.2)
    a = (a - a.mean()) * rng.uniform(0.8, 1.2) + a.mean()
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


def main() -> int:
    root = storage_path("raw", "ign_bdortho")
    meta = json.loads((root / "charolles_pasture_z19.json").read_text())
    img = cv2.imread(str(root / meta["image"]), cv2.IMREAD_GRAYSCALE)
    mpp = float(meta["metres_per_pixel"])
    step_x = FOOT_W_M * (1 - SIDE_OVERLAP)
    step_y = FOOT_H_M * (1 - FWD_OVERLAP)
    keyframes = []
    for i, e in enumerate(np.arange(-AREA_M / 2, AREA_M / 2 + 1e-6, step_x)):
        lane = np.arange(-AREA_M / 2, AREA_M / 2 + 1e-6, step_y)
        heading = 0.0 if i % 2 == 0 else math.pi
        for n in lane if i % 2 == 0 else lane[::-1]:
            f, to_m = frame(img, mpp, e, n, heading)
            keyframes.append(Keyframe(f, to_m))
    rng = np.random.default_rng(SEED)
    rows = []
    for k in range(N_IN + N_OUT):
        inside = k < N_IN
        if inside:
            e, n = rng.uniform(-AREA_M / 2, AREA_M / 2, 2)
        else:
            r, a = rng.uniform(AREA_M / 2 + 100, AREA_M / 2 + 200), rng.uniform(0, 2 * math.pi)
            e, n = r * math.cos(a), r * math.sin(a)
        heading = rng.uniform(0, 2 * math.pi)
        scale = HEIGHT_M / (HEIGHT_M + rng.uniform(-3, 3))
        f, to_m = frame(img, mpp, e, n, heading, scale)
        truth = to_m @ np.array([OUT_W / 2, OUT_H / 2, 1.0])
        ang = rng.uniform(0, 2 * math.pi)
        prior = truth + rng.uniform(0, 50) * np.array([math.cos(ang), math.sin(ang)])
        fix = locate(perturb(f, rng), keyframes, prior)
        err = None if fix is None else float(np.linalg.norm(fix.east_north - truth))
        rows.append(
            {
                "inside": inside,
                "fixed": fix is not None,
                "error_m": err,
                "inliers": None if fix is None else fix.inliers,
            }
        )
    ins = [r for r in rows if r["inside"]]
    errs = np.array([r["error_m"] for r in ins if r["fixed"]])
    wrong = sum(1 for r in rows if r["fixed"] and r["error_m"] > 20)
    out = {
        "note": "Imagery-only study on one real orthophoto; teach and repeat from the same image. "
        "Necessary, not sufficient; not a flight result.",
        "keyframes": len(keyframes),
        "inside_frames": len(ins),
        "inside_fixed_share": float(np.mean([r["fixed"] for r in ins])),
        "inside_error_median_m": float(np.median(errs)) if len(errs) else None,
        "inside_error_p95_m": float(np.percentile(errs, 95)) if len(errs) else None,
        "outside_frames": N_OUT,
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
    dest = REPO_ROOT / "results" / "phase5" / "visual_fix_study.json"
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
