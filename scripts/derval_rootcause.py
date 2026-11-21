"""Why does the Phase 1 detector miss most cattle on Derval? (Phase 2 plan, step C1)

Usage: python scripts/derval_rootcause.py [--detector dfine]

For every labelled animal in the cross-validation images, records whether the Phase 1 model
found it (IoU >= the evaluation threshold at its fold's operating threshold, as in
scripts/score_baseline.py), the best score of any prediction overlapping it, and candidate causes:
camera, altitude, ground sample distance (GSD), size in pixels and in metres, brightness of the
animal and of its surroundings, the share of very dark and very light pixels (coat pattern), and
whether it touches another animal.

Then compares Derval with the other farms:
  * miss rate per stratum (camera, altitude, size, brightness, pattern);
  * matched comparison: each Derval animal is paired with non-Derval animals of similar size in
    pixels and similar brightness; if Derval still misses far more, size and light do not
    explain it.

Writes results/baselines/derval_rootcause.json. Diagnostic only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from herdsight.data.icaerus import default_root, iter_images
from herdsight.data.splits import FlightStats
from herdsight.evaluate.upright_ap import iou_matrix
from herdsight.geo.nadir import focal_px_from_35mm, ground_sample_distance_m
from herdsight.utils.config import REPO_ROOT, load_config
from herdsight.utils.paths import storage_path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_baseline import cross_fit, load_preds  # noqa: E402

DARK, LIGHT = 60, 190  # grey levels for "very dark" and "very light" pixels


def box_stats(grey: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    """Mean brightness inside the box, of a ring around it, and dark/light pixel shares inside."""
    h, w = grey.shape
    x0, y0, x1, y1 = (int(round(v)) for v in b)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    inside = grey[y0:y1, x0:x1]
    pad = max(4, int(0.5 * max(x1 - x0, y1 - y0)))
    X0, Y0, X1, Y1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, w), min(y1 + pad, h)
    ring = grey[Y0:Y1, X0:X1].astype(float).copy()
    ring[y0 - Y0 : y1 - Y0, x0 - X0 : x1 - X0] = np.nan
    if inside.size == 0:
        return math.nan, math.nan, math.nan, math.nan
    return (
        float(inside.mean()),
        float(np.nanmean(ring)),
        float((inside < DARK).mean()),
        float((inside > LIGHT).mean()),
    )


def rate(flags: list[bool]) -> dict:
    return {"n": len(flags), "miss_rate": round(float(np.mean(flags)), 4) if flags else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="dfine")
    args = ap.parse_args(argv)
    ev = load_config("evaluation")
    iou_t = float(ev.require("iou_match_threshold"))
    sp = json.loads((REPO_ROOT / "results" / "splits.json").read_text())["icaerus_v2"]
    folds = [f"fold{k}" for k in range(sp["n_folds"])]
    preds: dict = {}
    for f in folds:
        preds |= load_preds(storage_path("runs", "baselines", args.detector) / f"{f}.jsonl")
    truth, fold_of, group_of, path_of = {}, {}, {}, {}
    for s, problem in iter_images(default_root(), with_metadata=False):
        fold = sp["flights"].get(f"{s.farm}/{s.flight}")
        if problem is None and s.labelled and fold in folds:
            k = f"{s.farm}/{s.flight}/{s.path.name}"
            truth[k], fold_of[k], path_of[k] = s.boxes_xyxy, fold, s.path
            group_of[k] = FlightStats(s.farm, s.flight, 0, 0).group
    meta = {}
    with (REPO_ROOT / "results" / "data" / "icaerus_images.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            meta[f"{r['farm']}/{r['flight']}/{r['image']}"] = r
    keys = sorted(k for k in preds if k in truth)
    start, stop, step = ev.require("operating_threshold_grid")
    grid = np.round(np.arange(start, stop + 1e-9, step), 4)
    fits = cross_fit(keys, fold_of, preds, truth, grid)

    rows = []
    for k in keys:
        t = truth[k]
        if not len(t):
            continue
        boxes, scores = preds[k]
        keep = scores >= fits[fold_of[k]]["threshold"]
        m_kept = iou_matrix(boxes[keep], t)
        found = np.zeros(len(t), bool)
        if m_kept.size:
            w = np.where(m_kept >= iou_t, m_kept, 0.0)
            r, c = linear_sum_assignment(w, maximize=True)
            found[c[w[r, c] >= iou_t]] = True
        m_all = iou_matrix(boxes, t)
        md = meta[k]
        width, height = int(md["width"]), int(md["height"])
        alt = float(md["rel_alt_m"]) if md["rel_alt_m"] else math.nan
        f_px = focal_px_from_35mm(float(md["focal_35mm"]), width, height)
        gsd = ground_sample_distance_m(alt, f_px)
        grey = cv2.imread(str(path_of[k]), cv2.IMREAD_GRAYSCALE)
        tt = iou_matrix(t, t)
        np.fill_diagonal(tt, 0.0)
        farm = k.split("/")[0]
        for j, b in enumerate(t):
            hit = m_all[:, j] >= iou_t if m_all.size else np.zeros(0, bool)
            long_px = float(max(b[2] - b[0], b[3] - b[1]))
            mean_in, mean_ring, dark, light = box_stats(grey, b)
            rows.append(
                {
                    "farm": farm,
                    "flight": k.split("/")[1],
                    "group": group_of[k],
                    "camera": md["camera_model"],
                    "alt_m": round(alt / 10) * 10 if math.isfinite(alt) else None,
                    "gsd_cm": 100 * gsd,
                    "long_px": long_px,
                    "long_m": long_px * gsd,
                    "brightness": mean_in,
                    "surround": mean_ring,
                    "dark_share": dark,
                    "light_share": light,
                    "touching": bool((tt[j] > 0).any()),
                    "missed": not bool(found[j]),
                    "best_score": float(scores[hit].max()) if hit.any() else 0.0,
                }
            )

    out: dict = {
        "detector": args.detector,
        "note": "Diagnostic only. Public-data results, not field results.",
        "n_animals": len(rows),
    }
    der = [r for r in rows if r["farm"] == "Derval"]
    oth = [r for r in rows if r["farm"] != "Derval"]

    def strat(key: str, rs: list[dict]) -> dict:
        g = defaultdict(list)
        for r in rs:
            g[str(r[key])].append(r["missed"])
        return {k: rate(v) for k, v in sorted(g.items())}

    out["by_farm_camera_altitude"] = {
        f"{r_farm}/{cam}/{alt}m": rate(
            [
                r["missed"]
                for r in rows
                if (r["farm"], r["camera"], r["alt_m"]) == (r_farm, cam, alt)
            ]
        )
        for r_farm, cam, alt in sorted(
            {(r["farm"], r["camera"], r["alt_m"]) for r in rows}, key=str
        )
    }
    out["derval_by_flight"] = strat("flight", der)

    def describe(rs: list[dict], key: str) -> dict:
        v = np.array([r[key] for r in rs], float)
        v = v[np.isfinite(v)]
        return {
            "median": round(float(np.median(v)), 3),
            "p10": round(float(np.quantile(v, 0.1)), 3),
            "p90": round(float(np.quantile(v, 0.9)), 3),
        }

    feats = ("gsd_cm", "long_px", "long_m", "brightness", "surround", "dark_share", "light_share")
    out["features"] = {
        name: {f: describe(rs, f) for f in feats}
        for name, rs in (
            ("derval", der),
            ("mauron_m3t", [r for r in oth if r["farm"] == "Mauron" and r["camera"] == "M3T"]),
            ("other_farms", oth),
        )
    }
    out["touching_share"] = {
        "derval": round(float(np.mean([r["touching"] for r in der])), 4),
        "other": round(float(np.mean([r["touching"] for r in oth])), 4),
    }

    # Miss rate by pixel size band, Derval vs the rest (is it just small animals?)
    bands = [0, 40, 60, 80, 100, 130, 1e9]
    out["miss_rate_by_long_px"] = {
        f"{lo:.0f}-{hi:.0f}" if hi < 1e8 else f"{lo:.0f}+": {
            "derval": rate([r["missed"] for r in der if lo <= r["long_px"] < hi]),
            "other": rate([r["missed"] for r in oth if lo <= r["long_px"] < hi]),
        }
        for lo, hi in zip(bands[:-1], bands[1:], strict=True)
    }
    # ...by brightness of the surroundings (shade)
    sb = [0, 60, 90, 120, 255]
    out["miss_rate_by_surround_brightness"] = {
        f"{lo}-{hi}": {
            "derval": rate([r["missed"] for r in der if lo <= r["surround"] < hi]),
            "other": rate([r["missed"] for r in oth if lo <= r["surround"] < hi]),
        }
        for lo, hi in zip(sb[:-1], sb[1:], strict=True)
    }

    # ...by coat pattern: animals with both dark and light patches (black-and-white)
    def pied(r: dict) -> bool:
        return r["dark_share"] > 0.15 and r["light_share"] > 0.15

    out["miss_rate_by_pied_pattern"] = {
        "derval": {
            "pied": rate([r["missed"] for r in der if pied(r)]),
            "not_pied": rate([r["missed"] for r in der if not pied(r)]),
        },
        "other": {
            "pied": rate([r["missed"] for r in oth if pied(r)]),
            "not_pied": rate([r["missed"] for r in oth if not pied(r)]),
        },
    }

    # Matched comparison on size in pixels and surrounding brightness
    oth_px = np.array([r["long_px"] for r in oth])
    oth_sur = np.array([r["surround"] for r in oth])
    oth_miss = np.array([r["missed"] for r in oth], float)
    matched, matched_der, used = [], [], 0
    for r in der:
        sel = (np.abs(oth_px - r["long_px"]) <= 0.15 * r["long_px"]) & (
            np.abs(oth_sur - r["surround"]) <= 15
        )
        if sel.sum() >= 5:
            matched.append(oth_miss[sel].mean())
            matched_der.append(r["missed"])
            used += 1
    out["matched_on_size_and_surround"] = {
        "derval_animals_matched": used,
        "derval_miss_rate_of_matched": round(float(np.mean(matched_der)), 4) if used else None,
        "expected_miss_rate_from_similar_other_animals": round(float(np.mean(matched)), 4)
        if matched
        else None,
    }

    def split(cond: Callable[[dict], bool]) -> dict:
        return {
            "derval": {
                "yes": rate([r["missed"] for r in der if cond(r)]),
                "no": rate([r["missed"] for r in der if not cond(r)]),
            },
            "other": {
                "yes": rate([r["missed"] for r in oth if cond(r)]),
                "no": rate([r["missed"] for r in oth if not cond(r)]),
            },
        }

    out["miss_rate_split"] = {
        "touching": split(lambda r: r["touching"]),
        "dark_animal (over 40% very dark pixels)": split(lambda r: r["dark_share"] > 0.4),
        "short_under_1.2m (partly visible or small)": split(lambda r: r["long_m"] < 1.2),
    }
    apart_light = [r["missed"] for r in der if not r["touching"] and r["dark_share"] <= 0.4]
    out["derval_apart_and_not_dark"] = rate(apart_light)
    sc_d = np.array([r["best_score"] for r in der])
    m3t = [r for r in oth if r["farm"] == "Mauron" and r["camera"] == "M3T"]
    sc_m = np.array([r["best_score"] for r in m3t])
    edges = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
    out["best_score_histogram"] = {
        "edges": edges,
        "derval": np.histogram(sc_d, edges)[0].tolist(),
        "mauron_m3t": np.histogram(sc_m, edges)[0].tolist(),
    }
    dest = REPO_ROOT / "results" / "baselines" / "derval_rootcause.json"
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
