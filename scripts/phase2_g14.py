"""G14: count ranges for a new farm, fitted on photos with animals and growing with the count (H28).

Usage: python scripts/phase2_g14.py --lofo dfine_g6ref --out results/phase2/g14_ranges.json

Written after the held-back farm (D-54) and judged on cross-validation only; the test farm is not
used. Every count comes from a leave-one-farm-out model (each farm counted by a model that never saw
it). For each farm, the operating threshold and the range quantiles come from the other two farms'
counts, and the farm's own counts are scored, so nothing from a farm sets its own range.

Two methods, both rounded outwards to whole animals and clipped at zero:
* current: one quantile of |n - c| / sqrt(c + 1) for all photos (the Phase 1 method);
* g14: photos with a predicted count below 1 keep the current method, fitted on such photos only;
  photos with a predicted count of 1 or more get [c + q_lo (c + 1), c + q_hi (c + 1)], where q_lo
  and q_hi are the conformal quantiles of the signed score (n - c) / (c + 1) at (1 - p) / 2 and
  (1 + p) / 2, fitted on such photos only.
Coverage is pooled over the three farms, overall and per band of the true count, with a 95%
interval from resampling farm-date groups. Public-data results, not field results.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from herdsight.data.icaerus import default_root, iter_images
from herdsight.data.splits import FlightStats
from herdsight.evaluate.conformal import conformal_quantile
from herdsight.evaluate.counts import count_interval, fit_count_conformal, predicted_band
from herdsight.evaluate.stats import group_bootstrap_ci
from herdsight.utils.config import REPO_ROOT, load_config
from herdsight.utils.env_report import env_report
from herdsight.utils.paths import storage_path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_baseline import LEVELS, best_threshold, load_preds  # noqa: E402

FARMS = ("Derval", "Jalogny", "Mauron")
METHODS = ("current", "g14")


def lower_q(scores: np.ndarray, level: float) -> float:
    """Conformal lower quantile: minus the upper quantile of the negated scores."""
    return -conformal_quantile(-scores, level)


def fit_g14(c: np.ndarray, t: np.ndarray, levels: tuple[float, ...]) -> dict:
    occ = c >= 1
    empty_q = fit_count_conformal(c[~occ], t[~occ], levels) if (~occ).any() else None
    s = (t[occ] - c[occ]) / (c[occ] + 1.0)
    q = {}
    for lv in levels:
        tail = (1.0 + lv) / 2.0
        q[lv] = (lower_q(s, tail), conformal_quantile(s, tail)) if s.size else (-math.inf, math.inf)
    return {
        "empty_q": empty_q,
        "occupied_q": q,
        "n_empty": int((~occ).sum()),
        "n_occ": int(occ.sum()),
    }


def g14_interval(c: np.ndarray, model: dict, lv: float) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.zeros_like(c), np.full_like(c, np.inf)
    occ = c >= 1
    if (~occ).any() and model["empty_q"] is not None:
        lo[~occ], hi[~occ] = count_interval(c[~occ], model["empty_q"][lv])
    if occ.any():
        ql, qh = model["occupied_q"][lv]
        if math.isfinite(ql) and math.isfinite(qh):
            lo[occ] = np.maximum(np.floor(c[occ] + ql * (c[occ] + 1.0)), 0.0)
            hi[occ] = np.ceil(c[occ] + qh * (c[occ] + 1.0))
    return lo, hi


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lofo", default="dfine_g6ref")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results/phase2/g14_ranges.json")
    args = ap.parse_args(argv)
    ev, p2 = load_config("evaluation"), load_config("phase2")
    edges = [float(x) for x in p2.require("count_bands")]
    n_boot, seed = int(ev.require("n_boot")), int(ev.require("bootstrap_seed"))
    start, stop, step = ev.require("operating_threshold_grid")
    grid = np.round(np.arange(start, stop + 1e-9, step), 4)
    sp = json.loads((REPO_ROOT / "results" / "splits.json").read_text())["icaerus_v2"]
    folds = [f"fold{k}" for k in range(sp["n_folds"])]
    truth, group_of, farm_of = {}, {}, {}
    for s, problem in iter_images(default_root(), with_metadata=False):
        fold = sp["flights"].get(f"{s.farm}/{s.flight}")
        if problem is None and s.labelled and fold in folds:
            k = f"{s.farm}/{s.flight}/{s.path.name}"
            truth[k], farm_of[k] = s.boxes_xyxy, s.farm
            group_of[k] = FlightStats(s.farm, s.flight, 0, 0).group
    runs = storage_path("runs", "baselines")
    preds: dict = {}
    for farm in FARMS:
        for f in folds:
            part = load_preds(runs / f"{args.lofo}_{farm}" / f"{f}.jsonl")
            wrong = [k for k in part if farm_of.get(k, farm) != farm]
            if wrong:
                raise SystemExit(f"{farm}: predictions for another farm, e.g. {wrong[0]}")
            preds |= part
    keys = sorted(k for k in truth if k in preds)
    n = len(keys)
    t = np.array([truth[k].shape[0] for k in keys], float)
    c = np.zeros(n)
    res = {f"{m}_{b}_{lv}": np.zeros(n) for m in METHODS for b in ("lo", "hi") for lv in LEVELS}
    fits = {}
    for farm in FARMS:
        other = [k for k in keys if farm_of[k] != farm]
        own = np.array([i for i, k in enumerate(keys) if farm_of[k] == farm])
        thr = float(best_threshold(other, preds, truth, grid)[0])
        c_cal = np.array([(preds[k][1] >= thr).sum() for k in other], float)
        t_cal = np.array([truth[k].shape[0] for k in other], float)
        cur = fit_count_conformal(c_cal, t_cal, LEVELS)
        g14 = fit_g14(c_cal, t_cal, LEVELS)
        c_own = np.array([(preds[keys[i]][1] >= thr).sum() for i in own], float)
        c[own] = c_own
        for lv in LEVELS:
            lo, hi = count_interval(c_own, cur[lv])
            res[f"current_lo_{lv}"][own], res[f"current_hi_{lv}"][own] = lo, hi
            lo, hi = g14_interval(c_own, g14, lv)
            res[f"g14_lo_{lv}"][own], res[f"g14_hi_{lv}"][own] = lo, hi
        fits[farm] = {
            "threshold": thr,
            "calibration_images": len(other),
            "current_q": {str(lv): q for lv, q in cur.items()},
            "g14": {
                "n_empty_predicted": g14["n_empty"],
                "n_with_animals_predicted": g14["n_occ"],
                "empty_q": {str(lv): q for lv, q in (g14["empty_q"] or {}).items()},
                "occupied_q_lo_hi": {str(lv): list(v) for lv, v in g14["occupied_q"].items()},
            },
        }
    groups = np.array([group_of[k] for k in keys])
    tb = predicted_band(t, edges)
    names = ["0", "1-4", "5-19", "20+"]

    def cell(mask: np.ndarray, m: str, lv: float) -> dict:
        if not mask.any():
            return {"n": 0}
        lo, hi = res[f"{m}_lo_{lv}"][mask], res[f"{m}_hi_{lv}"][mask]
        inside = ((t[mask] >= lo) & (t[mask] <= hi)).astype(float)
        ci = group_bootstrap_ci(inside, groups[mask], np.mean, n_boot, np.random.default_rng(seed))
        w = hi - lo
        return {
            "n": int(mask.sum()),
            "coverage": float(inside.mean()),
            "coverage_ci95": list(ci),
            "mean_width": float(np.mean(w)) if np.all(np.isfinite(w)) else None,
        }

    five = t >= 5
    cov = {
        m: {
            str(lv): {
                "all": cell(np.ones(n, bool), m, lv),
                "5_or_more": cell(five, m, lv),
                "by_true_band": {nm: cell(tb == b, m, lv) for b, nm in enumerate(names)},
            }
            for lv in LEVELS
        }
        for m in METHODS
    }
    g, cu = cov["g14"]["0.9"], cov["current"]["0.9"]
    ratio = g["5_or_more"]["mean_width"] / cu["5_or_more"]["mean_width"]
    held = (
        0.85 <= g["5_or_more"]["coverage"] <= 0.95 and g["all"]["coverage"] >= 0.85 and ratio <= 2.0
    )
    result = {
        "note": (
            "Public-data results, not field results. Leave-one-farm-out counts; each farm's "
            "threshold and ranges fitted on the other two farms. The held-back test farm is not "
            "used."
        ),
        "lofo_prefix": args.lofo,
        "n_images": n,
        "n_5_or_more": int(five.sum()),
        "fits": fits,
        "coverage": cov,
        "H28": {
            "rule": (
                "at 90%: G14 coverage on photos with 5 or more animals in [0.85, 0.95], overall "
                "coverage at least 0.85, mean width on 5 or more at most 2 times the current "
                "method's"
            ),
            "coverage_5_or_more": g["5_or_more"]["coverage"],
            "coverage_all": g["all"]["coverage"],
            "width_ratio_5_or_more": ratio,
            "held": bool(held),
        },
        "environment": env_report(),
    }
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"H28": result["H28"], "current_90": cu, "g14_90": g}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
