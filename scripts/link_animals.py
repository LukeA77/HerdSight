"""Candidate identities of labelled animals across overlapping images, for the S3 ground truth.

Usage: python scripts/link_animals.py

For every cross-validation flight (never the test farm):
  1. pairs of images whose geotag footprints overlap are listed;
  2. each pair is registered from image content (herdsight.geo.register, D-27), with the geotag
     prediction as a gate; if registration fails, the pair falls back to geotag projection;
  3. labelled animals in the two images are paired one-to-one (Hungarian) within a gate;
  4. links are grouped into candidate animals shortest first, never putting two labels from one
     image into one animal (herdsight.count.identities). Plain connected components are also
     counted, to show how much chaining the constraint prevents.

With --from-saved, step 4 is redone from the saved per-flight files (no registration).

These are CANDIDATES for hand verification (brief 8.3), not ground truth. Writes
results/data/s3_candidate_links.json (summary) and, per flight,
$HERDSIGHT_STORAGE/datasets/processed/s3_links/<farm>__<flight>.json (every link, for review).
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from herdsight.count.identities import constrained_identities
from herdsight.data.icaerus import default_root, iter_images
from herdsight.geo.nadir import camera_ned, focal_px_from_35mm, ground_to_pixel, pixel_to_ground
from herdsight.geo.register import apply_homography, features, register_features
from herdsight.utils.config import REPO_ROOT
from herdsight.utils.io import load_rgb
from herdsight.utils.paths import storage_path

GEOTAG_GATE_M = 15.0  # registration must agree with the geotag to within this
PIXEL_GATE_FRACTION = 0.5  # pairing gate: half the median animal length
GEOTAG_PAIR_GATE_M = 3.0  # pairing gate when falling back to geotags


def centres(b: np.ndarray) -> np.ndarray:
    return np.column_stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2])


class FeatureCache:
    """SIFT features of each image in a flight, computed once on first use."""

    def __init__(self, paths: list) -> None:
        self.paths, self.cache = paths, {}

    def __call__(self, i: int) -> object:
        if i not in self.cache:
            self.cache[i] = features(np.asarray(load_rgb(self.paths[i])))
        return self.cache[i]


class UnionFind:
    def __init__(self) -> None:
        self.p: dict = {}

    def find(self, x: object) -> object:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: object, b: object) -> None:
        self.p[self.find(a)] = self.find(b)


def group(detail: dict) -> dict:
    """Constrained identities, plus plain connected components for comparison."""
    counts = [len(b) for b in detail["boxes"]]
    links = [(tuple(lk["a"]), tuple(lk["b"]), lk["dist_m"]) for lk in detail["links"]]
    comps = constrained_identities(counts, links)
    uf = UnionFind()
    for a, b, _ in links:
        uf.union(a, b)
    plain: dict[object, list] = defaultdict(list)
    for i, n in enumerate(counts):
        for k in range(n):
            plain[uf.find((i, k))].append((i, k))
    return {
        "components": comps,
        "candidate_unique": len(comps),
        "unconstrained_unique": len(plain),
        "unconstrained_conflicts": sum(
            1 for v in plain.values() if len({i for i, _ in v}) < len(v)
        ),
        "max_in_one_image": max(counts) if counts else 0,
    }


def regroup_saved() -> int:
    links_dir = storage_path("processed", "s3_links")
    dest = REPO_ROOT / "results" / "data" / "s3_candidate_links.json"
    summary = json.loads(dest.read_text(encoding="utf-8"))
    for key in summary:
        farm, flight = key.split("/")
        path = links_dir / f"{farm}__{flight}.json"
        detail = json.loads(path.read_text(encoding="utf-8"))
        g = group(detail)
        detail["components"] = g.pop("components")
        path.write_text(json.dumps(detail), encoding="utf-8")
        summary[key].pop("conflict_components", None)
        summary[key].update(g)
        print(key, g, flush=True)
    dest.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    if "--from-saved" in sys.argv[1:]:
        return regroup_saved()
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
        m = meta[(s.farm, s.flight, s.path.name)]
        if m["lat"] and m["rel_alt_m"] and m["focal_35mm"] and m["gimbal_yaw_deg"]:
            flights[(s.farm, s.flight)].append((s, m))

    out_dir = storage_path("processed", "s3_links", create=True)
    summary = {}
    for (farm, flight), items in sorted(flights.items()):
        t0 = time.time()
        items.sort(key=lambda it: (it[1]["time"], it[0].path.name))
        lat0, lon0 = float(items[0][1]["lat"]), float(items[0][1]["lon"])
        cams, fpx, alt, yaw, foot = [], [], [], [], []
        for s, m in items:
            cams.append(camera_ned([float(m["lat"])], [float(m["lon"])], lat0, lon0)[0])
            fpx.append(focal_px_from_35mm(float(m["focal_35mm"]), s.width, s.height))
            alt.append(float(m["rel_alt_m"]))
            yaw.append(float(m["gimbal_yaw_deg"]))
            foot.append(min(s.width, s.height) * alt[-1] / fpx[-1])
        feat = FeatureCache([s.path for s, _ in items])

        links, pair_log = [], []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if np.linalg.norm(cams[i] - cams[j]) > 0.9 * min(foot[i], foot[j]):
                    continue  # footprints do not overlap enough to share animals
                si, sj = items[i][0], items[j][0]
                ci, cj = centres(si.boxes_xyxy), centres(sj.boxes_xyxy)
                d_ne = cams[j] - cams[i]
                prior = ground_to_pixel(
                    [d_ne[0]], [d_ne[1]], alt[i], fpx[i], yaw[i], si.width, si.height
                )[0]
                reg = register_features(
                    feat(i),
                    feat(j),
                    prior_b_centre_in_a=prior,
                    max_disagreement_px=GEOTAG_GATE_M * fpx[i] / alt[i],
                )
                if reg.accepted:
                    cj_in_i = apply_homography(reg.H_ab, cj)
                    dist = np.linalg.norm(ci[:, None] - cj_in_i[None], axis=2)
                    lengths = np.concatenate(
                        [
                            np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
                            for b in (si.boxes_xyxy, sj.boxes_xyxy)
                        ]
                    )
                    gate = PIXEL_GATE_FRACTION * float(np.median(lengths))
                    m_per_px = alt[i] / fpx[i]
                    method = "registration"
                else:
                    gi = (
                        pixel_to_ground(
                            ci[:, 0], ci[:, 1], alt[i], fpx[i], yaw[i], si.width, si.height
                        )
                        + cams[i]
                    )
                    gj = (
                        pixel_to_ground(
                            cj[:, 0], cj[:, 1], alt[j], fpx[j], yaw[j], sj.width, sj.height
                        )
                        + cams[j]
                    )
                    dist = np.linalg.norm(gi[:, None] - gj[None], axis=2)
                    gate, m_per_px, method = GEOTAG_PAIR_GATE_M, 1.0, "geotag"
                r, c = linear_sum_assignment(np.where(dist <= gate, dist, 1e9))
                ok = dist[r, c] <= gate
                for a, b, d in zip(r[ok], c[ok], dist[r, c][ok], strict=True):
                    links.append(
                        {
                            "a": [i, int(a)],
                            "b": [j, int(b)],
                            "dist_m": float(d * m_per_px),
                            "method": method,
                        }
                    )
                pair_log.append(
                    {
                        "i": i,
                        "j": j,
                        "method": method,
                        "reason": reg.reason,
                        "inliers": reg.inliers,
                        "reprojection_px": reg.reprojection_px,
                        "H_ab": reg.H_ab.tolist() if reg.accepted else None,
                        "links": int(ok.sum()),
                    }
                )
        detail = {
            "farm": farm,
            "flight": flight,
            "images": [s.path.name for s, _ in items],
            "boxes": [s.boxes_xyxy.round(1).tolist() for s, _ in items],
            "pairs": pair_log,
            "links": links,
        }
        g = group(detail)
        detail["components"] = g.pop("components")
        reg_pairs = [p for p in pair_log if p["method"] == "registration"]
        reg_d = [lk["dist_m"] for lk in links if lk["method"] == "registration"]
        summary[f"{farm}/{flight}"] = {
            "images_with_animals": len(items),
            "naive_sum": sum(len(b) for b in detail["boxes"]),
            **g,
            "overlapping_pairs": len(pair_log),
            "registered_pairs": len(reg_pairs),
            "geotag_fallback_pairs": len(pair_log) - len(reg_pairs),
            "links": len(links),
            "registered_link_distance_m_median": float(np.median(reg_d)) if reg_d else None,
            "registered_link_distance_m_p90": float(np.percentile(reg_d, 90)) if reg_d else None,
            "seconds": round(time.time() - t0, 1),
        }
        (out_dir / f"{farm}__{flight}.json").write_text(json.dumps(detail), encoding="utf-8")
        print(f"{farm}/{flight}: {json.dumps(summary[f'{farm}/{flight}'])}", flush=True)
    dest = REPO_ROOT / "results" / "data" / "s3_candidate_links.json"
    dest.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
