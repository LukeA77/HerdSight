"""How far animals move between overlapping stills, from the hand-checked ICAERUS flights.

Usage: python scripts/herd_motion_stats.py

Reads the animal links of the S3 flights ($HERDSIGHT_STORAGE/datasets/processed/s3_links/,
written by scripts/link_animals.py; cross-validation farms only). For every link made through
image registration (D-27), the distance in metres between the animal's two positions is its
movement plus registration error, over the time between the two stills (from the DJI file
names). Writes results/data/herd_motion.json: distances by time gap, and a fit of the mean
squared distance as floor + 4 D t, where the floor is registration and box-centre error and D is
the grazing diffusion coefficient (2-D random walk), by least squares over all links.

Two limits, stated in the output:
  * links were only made within a gate of half a median animal length, so larger moves are
    missing (the distances are cut off at about 1 m);
  * registration and box-centre error are in the floor term, not in D; the cut-off at the gate
    makes D an underestimate once moves approach the gate (gaps of minutes).
This is the seconds-to-minutes timescale the simulator's herd model needs (Phase 5 plan, step 1).
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime

import numpy as np

from herdsight.utils.config import REPO_ROOT
from herdsight.utils.paths import storage_path

TIME = re.compile(r"DJI_(\d{14})_")
BINS_S = (0, 10, 30, 60, 120, 600)


def shot_time(name: str) -> datetime:
    m = TIME.search(name)
    if not m:
        raise ValueError(f"no time in {name}")
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")


def main() -> int:
    folder = storage_path("processed", "s3_links")
    rows = []
    for path in sorted(folder.glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        times = [shot_time(n) for n in d["images"]]
        for link in d["links"]:
            if link["method"] != "registration":
                continue
            i, j = link["a"][0], link["b"][0]
            dt = abs((times[j] - times[i]).total_seconds())
            rows.append((f"{d['farm']}/{d['flight']}", dt, float(link["dist_m"])))
    if not rows:
        raise SystemExit("no registration links found")
    dt = np.array([r[1] for r in rows])
    dist = np.array([r[2] for r in rows])
    ok = dt > 0
    design = np.column_stack([np.ones(ok.sum()), 4 * dt[ok]])
    (floor, diffusion), *_ = np.linalg.lstsq(design, dist[ok] ** 2, rcond=None)
    bins = []
    for lo, hi in zip(BINS_S[:-1], BINS_S[1:], strict=True):
        m = (dt > lo) & (dt <= hi)
        if m.sum():
            bins.append(
                {
                    "dt_s": [lo, hi],
                    "links": int(m.sum()),
                    "median_m": float(np.median(dist[m])),
                    "p90_m": float(np.quantile(dist[m], 0.9)),
                    "rms_m": float(np.sqrt(np.mean(dist[m] ** 2))),
                }
            )
    out = {
        "flights": sorted({r[0] for r in rows}),
        "links": len(rows),
        "dt_s_median": float(np.median(dt)),
        "distance_m": {
            "median": float(np.median(dist)),
            "p90": float(np.quantile(dist, 0.9)),
            "max": float(dist.max()),
        },
        "by_time_gap": bins,
        "fit": "mean squared distance = floor + 4 D t (least squares over links)",
        "floor_rms_m": float(np.sqrt(max(floor, 0.0))),
        "grazing_diffusion_m2_per_s": float(diffusion),
        "limits": [
            "links only within a gate of half a median animal length: larger moves are missing",
            "registration error is in the floor term; the cut-off at the gate makes D an under"
            "estimate for gaps of minutes, when moves approach the gate",
        ],
        "source": "S3 flights, cross-validation farms only (results/data/s3_truth.json)",
    }
    dest = REPO_ROOT / "results" / "data" / "herd_motion.json"
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
