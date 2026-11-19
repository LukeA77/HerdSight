"""Inventory of the ICAERUS v2 dataset: metadata, labels, overlap and animal sizes per flight.

Usage: python scripts/inspect_icaerus.py [--root <unpacked dataset>]
   (default root: the unpacked archive in the storage folder, see herdsight.data.icaerus)

Writes (small, committed):
  results/data/icaerus_inventory.json   per-flight and overall summary
  results/data/icaerus_images.csv       one row per image (metadata, box count, problems)

Ground overlap between consecutive images uses the camera footprint from the relative altitude
and the 35 mm-equivalent focal length (diagonal field of view of a 43.27 mm diagonal), on flat
ground below a nadir camera; GPS positions come from EXIF. It is an estimate for planning the
duplicate-removal benchmark, not a measured overlap.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from herdsight.data.icaerus import default_root, iter_images, unmatched_labels
from herdsight.utils.config import REPO_ROOT
from herdsight.utils.env_report import env_report

FULL_FRAME_DIAGONAL_MM = 43.27  # 36 x 24 mm
EARTH_RADIUS_M = 6_371_008.8  # mean radius (IUGG)


def footprint_m(rel_alt: float, f35: float, width: int, height: int) -> tuple[float, float]:
    """Ground width and height (m) seen by a nadir camera at rel_alt above flat ground."""
    diag_fov = 2 * math.atan(FULL_FRAME_DIAGONAL_MM / (2 * f35))
    diag_ground = 2 * rel_alt * math.tan(diag_fov / 2)
    k = diag_ground / math.hypot(width, height)
    return width * k, height * k


def ground_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance on a local tangent plane (accurate to well under 1% over a few hundred metres)."""
    phi = math.radians((lat1 + lat2) / 2)
    dn = math.radians(lat2 - lat1) * EARTH_RADIUS_M
    de = math.radians(lon2 - lon1) * EARTH_RADIUS_M * math.cos(phi)
    return math.hypot(dn, de)


def pct(a: list[float] | np.ndarray, q: float) -> float | None:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if a.size else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=None)
    args = ap.parse_args(argv)
    root = args.root or default_root()
    if not root.is_dir():
        print(f"dataset folder not found: {root}")
        return 1
    rows, problems = [], defaultdict(int)
    for s, problem in iter_images(root):
        if problem:
            problems[problem] += 1
        b = s.boxes_xyxy if s.labelled else np.zeros((0, 4))
        w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
        rows.append(
            {
                "farm": s.farm,
                "flight": s.flight,
                "image": s.path.name,
                "width": s.width,
                "height": s.height,
                "camera_model": s.camera_model,
                "focal_35mm": s.focal_length_35mm,
                "time": s.time,
                "lat": s.lat,
                "lon": s.lon,
                "rel_alt_m": s.rel_alt_m,
                "gimbal_pitch_deg": s.gimbal_pitch_deg,
                "gimbal_yaw_deg": s.gimbal_yaw_deg,
                "n_boxes": int(b.shape[0]) if s.labelled else None,
                "box_long_px_max": float(np.maximum(w, h).max()) if b.size else None,
                "problem": problem,
                "_long": np.maximum(w, h).tolist(),
                "_short": np.minimum(w, h).tolist(),
            }
        )
    if not rows:
        print("no images found")
        return 1

    flights: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        flights[(r["farm"], r["flight"])].append(r)
    per_flight = []
    for (farm, flight), rs in sorted(flights.items()):
        rs.sort(key=lambda r: (r["time"] or "", r["image"]))
        alts = [r["rel_alt_m"] for r in rs if r["rel_alt_m"] is not None]
        overlaps = []
        for a, b in zip(rs, rs[1:], strict=False):
            if None in (a["lat"], b["lat"], a["rel_alt_m"], a["focal_35mm"]):
                continue
            fw, fh = footprint_m(a["rel_alt_m"], a["focal_35mm"], a["width"], a["height"])
            d = ground_distance_m(a["lat"], a["lon"], b["lat"], b["lon"])
            overlaps.append(max(0.0, 1 - d / fh) if d > 0 else 1.0)  # along the short side
        longs = [x for r in rs for x in r["_long"]]
        shorts = [x for r in rs for x in r["_short"]]
        labelled = [r for r in rs if r["n_boxes"] is not None]
        per_flight.append(
            {
                "farm": farm,
                "flight": flight,
                "n_images": len(rs),
                "n_labelled": len(labelled),
                "n_boxes": sum(r["n_boxes"] for r in labelled),
                "n_empty_images": sum(1 for r in labelled if r["n_boxes"] == 0),
                "max_boxes_in_image": max((r["n_boxes"] for r in labelled), default=0),
                "cameras": sorted({r["camera_model"] or "?" for r in rs}),
                "image_sizes": sorted({f"{r['width']}x{r['height']}" for r in rs}),
                "rel_alt_m_median": pct(alts, 50),
                "rel_alt_m_range": [min(alts), max(alts)] if alts else None,
                "gps_share": sum(r["lat"] is not None for r in rs) / len(rs),
                "nadir_share": sum(
                    r["gimbal_pitch_deg"] is not None and r["gimbal_pitch_deg"] <= -85 for r in rs
                )
                / len(rs),
                "consecutive_overlap_median": pct(overlaps, 50),
                "consecutive_overlap_share_above_0.2": (
                    float(np.mean(np.asarray(overlaps) > 0.2)) if overlaps else None
                ),
                "box_long_px_p50": pct(longs, 50),
                "box_long_px_p99": pct(longs, 99),
                "box_long_px_max": max(longs) if longs else None,
                "box_short_px_p01": pct(shorts, 1),
                "box_short_px_p50": pct(shorts, 50),
                "time_first": rs[0]["time"],
            }
        )
    all_long = [x for r in rows for x in r["_long"]]
    all_short = [x for r in rows for x in r["_short"]]
    farms: dict[str, dict[str, int]] = defaultdict(lambda: {"flights": 0, "images": 0, "boxes": 0})
    for f in per_flight:
        farms[f["farm"]]["flights"] += 1
        farms[f["farm"]]["images"] += f["n_images"]
        farms[f["farm"]]["boxes"] += f["n_boxes"]
    inventory = {
        "dataset": "ICAERUS grazing cows v2 (Zenodo 10.5281/zenodo.11048412, CC BY 4.0)",
        "root": str(root),
        "n_images": len(rows),
        "n_boxes": int(sum(r["n_boxes"] or 0 for r in rows)),
        "n_flights": len(per_flight),
        "farms": dict(sorted(farms.items())),
        "problems": dict(problems),
        "label_files_without_image": unmatched_labels(root),
        "box_long_px": {q: pct(all_long, q) for q in (1, 50, 99, 99.9, 100)},
        "box_short_px": {q: pct(all_short, q) for q in (0, 1, 5, 50)},
        "flights": per_flight,
        "environment": env_report(),
    }
    out = REPO_ROOT / "results" / "data"
    out.mkdir(parents=True, exist_ok=True)
    (out / "icaerus_inventory.json").write_text(json.dumps(inventory, indent=1) + "\n")
    cols = [k for k in rows[0] if not k.startswith("_")]
    with (out / "icaerus_images.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        for r in sorted(rows, key=lambda r: (r["farm"], r["flight"], r["image"])):
            wr.writerow({k: r[k] for k in cols})
    print(
        json.dumps(
            {k: v for k, v in inventory.items() if k not in ("flights", "environment")}, indent=1
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
