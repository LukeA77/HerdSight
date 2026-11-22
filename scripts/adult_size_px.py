"""Size of labelled adult cattle in pixels on the cross-validation farms (Phase 5 planner input).

Usage: python scripts/adult_size_px.py

The mission planner needs a provisional pixel budget for the smallest animal to count (calves)
until the Phase 2 resolution study (G11) measures it. The provisional rule (Phase 5 plan, step
2): a calf must cover at least as many pixels as the smallest adults the detector was trained on,
taken as the 5th percentile of the long side of labelled boxes. Upright boxes of animals lying at
an angle are shorter than the animal on their long side, so this is a cautious (small) figure.
Also gives the long side in metres (height above take-off / focal length in pixels; wrong on
slopes), used for the size of simulated adults. Writes results/data/adult_size_px.json.
Test-farm images are never read.
"""

from __future__ import annotations

import csv
import json
import sys

import numpy as np

from herdsight.data.icaerus import default_root, iter_images
from herdsight.geo.nadir import focal_px_from_35mm
from herdsight.utils.config import REPO_ROOT

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def main() -> int:
    sp = json.loads((REPO_ROOT / "results" / "splits.json").read_text(encoding="utf-8"))
    flights = sp["icaerus_v2"]["flights"]
    meta = {
        f"{r['farm']}/{r['flight']}/{r['image']}": r
        for r in csv.DictReader((REPO_ROOT / "results" / "data" / "icaerus_images.csv").open())
    }
    metres: list[float] = []
    longs: list[float] = []
    per_height: dict[str, list[float]] = {}
    images = 0
    for s, problem in iter_images(default_root(), with_metadata=False):
        fold = flights.get(f"{s.farm}/{s.flight}")
        if not fold or not fold.startswith("fold") or problem or not s.labelled:
            continue
        b = np.asarray(s.boxes_xyxy, float).reshape(-1, 4)
        if not len(b):
            continue
        images += 1
        side = np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
        longs += side.tolist()
        m = meta.get(f"{s.farm}/{s.flight}/{s.path.name}", {})
        if m.get("rel_alt_m") and m.get("focal_35mm"):
            f_px = focal_px_from_35mm(float(m["focal_35mm"]), s.width, s.height)
            metres += (side * float(m["rel_alt_m"]) / f_px).tolist()
        per_height.setdefault(f"{s.width}x{s.height}", []).extend(side.tolist())
    longs_a = np.array(longs)
    out = {
        "images": images,
        "boxes": len(longs),
        "long_side_px_quantiles": {
            str(int(q * 100)): float(np.quantile(longs_a, q)) for q in QUANTILES
        },
        "long_side_m_quantiles": {
            str(int(q * 100)): float(np.quantile(metres, q)) for q in QUANTILES
        },
        "by_image_size": {
            k: {"boxes": len(v), "p5_px": float(np.quantile(v, 0.05))}
            for k, v in sorted(per_height.items())
        },
        "note": "long side of upright label boxes, cross-validation farms only; a cautious "
        "provisional pixel budget until the Phase 2 resolution study (G11)",
    }
    dest = REPO_ROOT / "results" / "data" / "adult_size_px.json"
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
