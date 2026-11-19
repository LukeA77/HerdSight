"""Spacing of labelled cattle on the ground, as evidence for the herd definition used by S7.

Usage: python scripts/herd_stats.py

For every labelled image in the cross-validation folds (never the test farm), label centres are
converted to ground metres with the image's ground sample distance (relative altitude and
35 mm-equivalent focal length, nadir camera). Reported: the nearest-neighbour distance between
animals in the same image, and the size of groups formed by single linkage at several distances
(animals closer than d metres are in the same group).

Writes results/data/herd_stats.json.
"""

from __future__ import annotations

import csv
import json
import sys

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist

from herdsight.data.icaerus import default_root, iter_images
from herdsight.geo.nadir import focal_px_from_35mm
from herdsight.utils.config import REPO_ROOT

LINK_DISTANCES_M = (5.0, 10.0, 20.0, 30.0)


def main() -> int:
    splits = json.loads((REPO_ROOT / "results" / "splits.json").read_text())["icaerus_v2"][
        "flights"
    ]
    meta = {}
    with (REPO_ROOT / "results" / "data" / "icaerus_images.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            meta[(r["farm"], r["flight"], r["image"])] = r
    nn, groups = [], {d: [] for d in LINK_DISTANCES_M}
    n_images = 0
    for s, problem in iter_images(default_root(), with_metadata=False):
        if problem or splits.get(f"{s.farm}/{s.flight}", "test") == "test" or not s.count:
            continue
        m = meta[(s.farm, s.flight, s.path.name)]
        if not m["rel_alt_m"] or not m["focal_35mm"]:
            continue
        n_images += 1
        gsd = float(m["rel_alt_m"]) / focal_px_from_35mm(float(m["focal_35mm"]), s.width, s.height)
        b = s.boxes_xyxy
        c = np.column_stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]) * gsd
        if len(c) >= 2:
            d = cdist(c, c)
            np.fill_diagonal(d, np.inf)
            nn += d.min(axis=1).tolist()
        for dist in LINK_DISTANCES_M:
            if len(c) == 1:
                groups[dist].append(1)
                continue
            dd = cdist(c, c)
            n, lab = connected_components(dd < dist, directed=False)
            groups[dist] += np.bincount(lab).tolist()
    nn_a = np.asarray(nn)
    out = {
        "images": n_images,
        "animals_with_a_neighbour_in_image": int(nn_a.size),
        "nearest_neighbour_m": {str(q): float(np.percentile(nn_a, q)) for q in (5, 25, 50, 75, 95)},
        "group_size_by_link_distance_m": {
            str(d): {
                "groups": len(g),
                "median_size": float(np.median(g)),
                "share_of_animals_in_groups_of_3_or_more": float(
                    np.sum([x for x in g if x >= 3]) / max(np.sum(g), 1)
                ),
                "largest": int(max(g)),
            }
            for d, g in groups.items()
        },
        "caveat": "groups stop at the edge of one image, so large herds are cut at image edges",
    }
    dest = REPO_ROOT / "results" / "data" / "herd_stats.json"
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
