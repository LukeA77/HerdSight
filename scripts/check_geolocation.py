"""Measure how well geotagged stills place the same animal at the same ground position.

Usage: python scripts/check_geolocation.py

For each pair of consecutive images in a flight (by capture time) that both contain labelled
cattle, every label centre is projected to the ground (herdsight.geo.nadir) from the image's GPS
position, relative altitude, 35 mm-equivalent focal length and gimbal yaw. Centres in the two
images are paired one-to-one by minimum total ground distance (Hungarian), within a gate. The
distances of the pairs measure the combined error of GPS, height, yaw and animal movement over the
seconds between images.

Several camera conventions are compared, so the orientation of the image on the ground is
measured rather than assumed: top of image facing the gimbal yaw, the flight yaw, or the reverse
of the gimbal yaw, each with and without a left-right mirror. The convention with the smallest
median distance is reported as the measured one.

Uses only the cross-validation folds (never the test farm).
Writes results/data/geolocation_check.json.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from herdsight.data.icaerus import default_root, iter_images
from herdsight.geo.nadir import camera_ned, focal_px_from_35mm, pixel_to_ground
from herdsight.utils.config import REPO_ROOT

GATE_M = 15.0
CONVENTIONS = {
    "gimbal_yaw": ("gimbal_yaw_deg", 0.0, False),
    "gimbal_yaw_mirrored": ("gimbal_yaw_deg", 0.0, True),
    "gimbal_yaw_plus_180": ("gimbal_yaw_deg", 180.0, False),
    "flight_yaw": ("flight_yaw_deg", 0.0, False),
}


def main() -> int:
    splits = json.loads((REPO_ROOT / "results" / "splits.json").read_text())["icaerus_v2"][
        "flights"
    ]
    meta = {}
    with (REPO_ROOT / "results" / "data" / "icaerus_images.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            meta[(r["farm"], r["flight"], r["image"])] = r
    flights: dict[tuple[str, str], list] = defaultdict(list)
    for s, problem in iter_images(default_root(), with_metadata=False):
        if problem or splits.get(f"{s.farm}/{s.flight}", "test") == "test" or not s.count:
            continue
        flights[(s.farm, s.flight)].append(s)

    # flight yaw is not in the inventory CSV; read it once per image when needed
    from herdsight.data.survey import read_survey_image

    results: dict[str, dict] = {}
    for name, (yaw_field, yaw_add, mirror) in CONVENTIONS.items():
        per_flight, all_d = {}, []
        for (farm, flight), imgs in sorted(flights.items()):
            rows = []
            for s in imgs:
                m = meta[(farm, flight, s.path.name)]
                if not m["lat"] or not m["rel_alt_m"] or not m["focal_35mm"]:
                    continue
                if yaw_field == "gimbal_yaw_deg":
                    yaw = float(m["gimbal_yaw_deg"])
                else:
                    yaw = read_survey_image(s.path).flight_yaw_deg
                    if yaw is None:
                        continue
                rows.append((m["time"], s, m, yaw + yaw_add))
            rows.sort(key=lambda r: (r[0], r[1].path.name))
            if len(rows) < 2:
                continue
            lat0, lon0 = float(rows[0][2]["lat"]), float(rows[0][2]["lon"])
            ground = []
            for _, s, m, yaw in rows:
                f_px = focal_px_from_35mm(float(m["focal_35mm"]), s.width, s.height)
                c = np.column_stack(
                    [
                        (s.boxes_xyxy[:, 0] + s.boxes_xyxy[:, 2]) / 2,
                        (s.boxes_xyxy[:, 1] + s.boxes_xyxy[:, 3]) / 2,
                    ]
                )
                off = pixel_to_ground(
                    c[:, 0], c[:, 1], float(m["rel_alt_m"]), f_px, yaw, s.width, s.height, mirror
                )
                cam = camera_ned([float(m["lat"])], [float(m["lon"])], lat0, lon0)[0]
                ground.append(off + cam)
            d_flight = []
            for a, b in zip(ground, ground[1:], strict=False):
                dist = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
                r, c = linear_sum_assignment(np.where(dist <= GATE_M, dist, 1e6))
                ok = dist[r, c] <= GATE_M
                d_flight += dist[r, c][ok].tolist()
            if d_flight:
                per_flight[f"{farm}/{flight}"] = {
                    "pairs": len(d_flight),
                    "median_m": float(np.median(d_flight)),
                    "p90_m": float(np.percentile(d_flight, 90)),
                }
                all_d += d_flight
        results[name] = {
            "pairs": len(all_d),
            "median_m": float(np.median(all_d)) if all_d else None,
            "p90_m": float(np.percentile(all_d, 90)) if all_d else None,
            "share_within_2m": float(np.mean(np.asarray(all_d) <= 2.0)) if all_d else None,
            "per_flight": per_flight,
        }
    best = min(results, key=lambda k: results[k]["median_m"] or 1e9)
    out = {
        "gate_m": GATE_M,
        "measured_convention": best,
        "conventions": results,
        "note": "distances combine GPS, height, yaw and animal movement between consecutive images",
    }
    dest = REPO_ROOT / "results" / "data" / "geolocation_check.json"
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: {kk: v[kk] for kk in ("pairs", "median_m", "p90_m", "share_within_2m")}
                for k, v in results.items()
            },
            indent=1,
        )
    )
    print("measured convention:", best)
    return 0


if __name__ == "__main__":
    sys.exit(main())
