"""S3 baselines: count per flight after removing duplicates across overlapping images.

Usage:
  python scripts/score_s3.py --detector dfine   (out-of-fold detections)
  python scripts/score_s3.py --oracle           (labels as detections: checks the pipeline)

For every flight in results/data/s3_candidate_links.json, on the images with at least one
labelled animal (the set the links were built on), three estimates are compared:
  naive         sum of per-image counts (what H6 says overcounts);
  geotag        detections projected to the ground from the geotag, paired within 3 m between
                overlapping images, grouped one-per-image (D-30);
  registration  pairs registered from image content (saved by scripts/link_animals.py),
                paired in pixels, falling back to geotags where registration failed, then
                grouped (D-27, D-30).
Truth is the owner-verified count (results/data/s3_truth.json) where it exists and is complete.
Other flights are scored against the unverified candidate count, and reported separately: that
comparison only shows agreement with the label-based linking, not accuracy.

Detection thresholds per fold follow scripts/score_baseline.py (count-MAE minimiser on the other
folds). False positives in images with no labelled animal are not included here; S2's empty band
measures them. Writes results/baselines/<detector>_s3.json (or oracle_s3.json).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from herdsight.count.identities import constrained_identities
from herdsight.count.link import match_points
from herdsight.data.icaerus import default_root, iter_images
from herdsight.geo.nadir import camera_ned, focal_px_from_35mm, pixel_to_ground
from herdsight.geo.register import apply_homography
from herdsight.utils.config import REPO_ROOT, load_config
from herdsight.utils.env_report import env_report
from herdsight.utils.paths import storage_path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from link_animals import GEOTAG_PAIR_GATE_M, PIXEL_GATE_FRACTION, centres  # noqa: E402
from score_baseline import cross_fit, load_preds  # noqa: E402

METHODS = ("naive", "geotag", "registration")


def flight_geometry(farm: str, flight: str, names: list[str], meta: dict, sizes: dict) -> dict:
    lat0, lon0 = (
        float(meta[(farm, flight, names[0])]["lat"]),
        float(meta[(farm, flight, names[0])]["lon"]),
    )
    g: dict[str, list] = {"cam": [], "fpx": [], "alt": [], "yaw": [], "wh": []}
    for n in names:
        m = meta[(farm, flight, n)]
        w, h = sizes[(farm, flight, n)]
        g["cam"].append(camera_ned([float(m["lat"])], [float(m["lon"])], lat0, lon0)[0])
        g["fpx"].append(focal_px_from_35mm(float(m["focal_35mm"]), w, h))
        g["alt"].append(float(m["rel_alt_m"]))
        g["yaw"].append(float(m["gimbal_yaw_deg"]))
        g["wh"].append((w, h))
    return g


def ground(g: dict, i: int, c: np.ndarray) -> np.ndarray:
    w, h = g["wh"][i]
    return (
        pixel_to_ground(c[:, 0], c[:, 1], g["alt"][i], g["fpx"][i], g["yaw"][i], w, h) + g["cam"][i]
    )


def estimate(detail: dict, boxes: list[np.ndarray], g: dict) -> dict[str, int]:
    counts = [len(b) for b in boxes]
    cs = [centres(b) if len(b) else np.zeros((0, 2)) for b in boxes]
    out = {"naive": int(sum(counts))}
    for method in ("geotag", "registration"):
        links = []
        for p in detail["pairs"]:
            i, j = p["i"], p["j"]
            if not counts[i] or not counts[j]:
                continue
            if method == "registration" and p["H_ab"] is not None:
                lengths = np.concatenate(
                    [np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]) for b in (boxes[i], boxes[j])]
                )
                gate = PIXEL_GATE_FRACTION * float(np.median(lengths))
                pairs = match_points(cs[i], apply_homography(np.asarray(p["H_ab"]), cs[j]), gate)
                scale = g["alt"][i] / g["fpx"][i]
            else:
                pairs = match_points(ground(g, i, cs[i]), ground(g, j, cs[j]), GEOTAG_PAIR_GATE_M)
                scale = 1.0
            links += [((i, a), (j, b), d * scale) for a, b, d in pairs]
        out[method] = len(constrained_identities(counts, links))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--detector")
    src.add_argument("--oracle", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    name = "oracle" if args.oracle else args.detector
    ev = load_config("evaluation")
    splits_path = REPO_ROOT / "results" / "splits.json"
    sp = json.loads(splits_path.read_text())["icaerus_v2"]
    folds = [f"fold{k}" for k in range(sp["n_folds"])]
    summary = json.loads((REPO_ROOT / "results" / "data" / "s3_candidate_links.json").read_text())
    truth_path = REPO_ROOT / "results" / "data" / "s3_truth.json"
    verified = json.loads(truth_path.read_text())["flights"] if truth_path.exists() else {}

    meta = {}
    with (REPO_ROOT / "results" / "data" / "icaerus_images.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            meta[(r["farm"], r["flight"], r["image"])] = r
    labels, sizes, fold_of = {}, {}, {}
    for s, problem in iter_images(default_root(), with_metadata=False):
        fold = sp["flights"].get(f"{s.farm}/{s.flight}")
        if problem is None and s.labelled and fold in folds:
            k = f"{s.farm}/{s.flight}/{s.path.name}"
            labels[k], fold_of[k] = s.boxes_xyxy, fold
            sizes[(s.farm, s.flight, s.path.name)] = (s.width, s.height)

    thresholds: dict[str, float] = {}
    if args.oracle:
        preds = {k: (b, np.ones(len(b))) for k, b in labels.items()}
        thresholds = dict.fromkeys(folds, 0.5)
    else:
        preds = {}
        runs = storage_path("runs", "baselines", args.detector)
        for f in folds:
            preds |= load_preds(runs / f"{f}.jsonl")
        keys = sorted(k for k in preds if k in labels)
        start, stop, step = ev.require("operating_threshold_grid")
        grid = np.round(np.arange(start, stop + 1e-9, step), 4)
        thresholds = {
            k: v["threshold"] for k, v in cross_fit(keys, fold_of, preds, labels, grid).items()
        }

    links_dir = storage_path("processed", "s3_links")
    flights, skipped = {}, []
    for key in sorted(summary):
        farm, flight = key.split("/")
        detail = json.loads((links_dir / f"{farm}__{flight}.json").read_text(encoding="utf-8"))
        ks = [f"{key}/{n}" for n in detail["images"]]
        if any(k not in preds for k in ks):
            skipped.append(key)
            continue
        boxes = [preds[k][0][preds[k][1] >= thresholds[fold_of[k]]] for k in ks]
        g = flight_geometry(farm, flight, detail["images"], meta, sizes)
        est = estimate(detail, boxes, g)
        v = verified.get(key)
        is_verified = bool(v and v["complete"])
        true_n = v["verified_unique"] if is_verified else summary[key]["candidate_unique"]
        flights[key] = {
            "fold": fold_of[ks[0]],
            "overlapping_pairs": len(detail["pairs"]),
            "truth": true_n,
            "truth_kind": "owner-verified" if is_verified else "unverified candidate",
            "estimates": est,
            "relative_error": {m: (est[m] - true_n) / true_n for m in METHODS},
        }

    def pooled(kind: str) -> dict | None:
        rows = [f for f in flights.values() if f["truth_kind"] == kind]
        if not rows:
            return None
        return {
            "flights": len(rows),
            **{
                f"{m}_mean_abs_relative_error": float(
                    np.mean([abs(r["relative_error"][m]) for r in rows])
                )
                for m in METHODS
            },
            # H6 needs overlap: a flight whose labelled images never overlap cannot overcount.
            "h6_flights_with_overlap": len(ov := [r for r in rows if r["overlapping_pairs"]]),
            "h6_naive_over_20pct_on_every_overlapping_flight": (
                all(r["relative_error"]["naive"] > 0.2 for r in ov) if ov else None
            ),
            "h6_naive_relative_error_min": (
                float(min(r["relative_error"]["naive"] for r in ov)) if ov else None
            ),
        }

    result = {
        "detector": name,
        "splits_sha256": hashlib.sha256(splits_path.read_bytes()).hexdigest(),
        "per_fold_threshold": thresholds,
        "verified": pooled("owner-verified"),
        "unverified": pooled("unverified candidate"),
        "flights": flights,
        "flights_skipped_missing_predictions": skipped,
        "environment": env_report(),
    }
    dest = args.out or REPO_ROOT / "results" / "baselines" / f"{name}_s3.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("verified", "unverified")}, indent=1))
    for key, f in flights.items():
        print(key, f["truth"], f["estimates"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
