"""Fit the geotag and registration error models used on simulated flights (Phase 5 plan, step 3).

Usage: python scripts/fit_flight_error_models.py

Geotags: results/data/geolocation_check.json gives, per real ICAERUS flight, the median distance
between where overlapping stills place the same ground point from their geotags alone. If each
still's geotag error is a 2-D Gaussian with standard deviation sigma, independent between stills,
the distance between two stills' errors has median 1.665 sigma (Rayleigh with scale sigma * sqrt 2).
So each flight gives sigma = median / 1.665; a simulated flight draws its sigma from these,
weighted by each real flight's number of pairs (as the pooled measurement is).

Registration (D-27): the S3 flights' overlapping pairs ($HERDSIGHT_STORAGE/datasets/processed/
s3_links/) give the share of pairs registered from image content, the residual in pixels of
those that were, and the reasons for falling back to the geotag.

Writes results/data/flight_error_model.json. Registration is modelled in simulation, never
tested there: rendered ground is not real pasture (Phase 5 plan).
"""

from __future__ import annotations

import json
import sys
from collections import Counter

import numpy as np

from herdsight.utils.config import REPO_ROOT
from herdsight.utils.paths import storage_path

RAYLEIGH_MEDIAN_OF_DIFFERENCE = float(np.sqrt(2) * np.sqrt(2 * np.log(2)))  # 1.665


def main() -> int:
    g = json.loads((REPO_ROOT / "results" / "data" / "geolocation_check.json").read_text())
    c = g["conventions"][g["measured_convention"]]
    per_flight = {k: v["median_m"] for k, v in c["per_flight"].items()}
    sigmas = {k: m / RAYLEIGH_MEDIAN_OF_DIFFERENCE for k, m in per_flight.items()}
    reasons: Counter = Counter()
    residual = []
    for path in sorted(storage_path("processed", "s3_links").glob("*.json")):
        for p in json.loads(path.read_text(encoding="utf-8"))["pairs"]:
            reasons[f"{p['method']}: {p['reason']}"] += 1
            if p["method"] == "registration":
                residual.append(float(p["reprojection_px"]))
    total = sum(reasons.values())
    accepted = sum(v for k, v in reasons.items() if k.startswith("registration"))
    out = {
        "geotag": {
            "model": "per-still 2-D Gaussian error, independent between stills; sigma per flight",
            "sigma_m_by_flight": sigmas,
            "pairs_by_flight": {k: v["pairs"] for k, v in c["per_flight"].items()},
            "measured_disagreement": {"median_m": c["median_m"], "p90_m": c["p90_m"]},
            "source": "results/data/geolocation_check.json (real ICAERUS flights)",
        },
        "registration": {
            "pairs": total,
            "accepted_share": accepted / total,
            "outcomes": dict(reasons),
            "residual_px": {
                "median": float(np.median(residual)),
                "p90": float(np.quantile(residual, 0.9)),
                "values": sorted(round(r, 3) for r in residual),
            },
            "source": "S3 flights' overlapping pairs (scripts/link_animals.py output)",
        },
        "note": "Real-flight measurements; in simulation registration is modelled, not tested",
    }
    dest = REPO_ROOT / "results" / "data" / "flight_error_model.json"
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                **out,
                "registration": {k: v for k, v in out["registration"].items() if k != "values"},
            },
            indent=1,
        )[:1500]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
