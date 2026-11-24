"""Phase 2 step G9: band-wise conformal count intervals, out of fold, with H11 (C3).

Usage: python scripts/phase2_g9.py --detector dfine_ref --lofo dfine_g6ref --out <json>

Per image, the interval around the predicted count c is c +/- q_p * sqrt(c + 1), rounded outwards
to whole animals (the frozen conformity score, configs/phase2.yaml). Two methods are compared:
* pooled: one q_p for all images (the Phase 1 method);
* band-wise: one q_p per band of the predicted count, with the frozen band edges (0, 1-4, 5-19,
  20+), so coverage holds within each band, not only on average.
Cross-validation: for each fold, the operating threshold and the quantiles come from the other two
folds' out-of-fold predictions. Those predictions were made by models that trained on the fold
being scored, so a small leakage remains (the Phase 1 method's); no nested models exist for every
fold, and this is stated in the result. Leave-one-farm-out (if --lofo is given): each farm's
predictions come from a model that never saw it, and the threshold and quantiles come from the
other two farms, so nothing from the farm sets its interval. This is where exchangeability is
weakest.
Coverage is reported at 68, 90 and 95% overall, per band of the true count and per band of the
predicted count, each with a 95% interval from resampling farm-date groups and the mean width
beside it. H11 is judged on images with 20 or more animals, at 90%. Public-data results, not
field results.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from herdsight.data.icaerus import default_root, iter_images
from herdsight.data.splits import FlightStats
from herdsight.evaluate.counts import (
    band_interval,
    count_interval,
    fit_band_conformal,
    fit_count_conformal,
    predicted_band,
)
from herdsight.evaluate.stats import group_bootstrap_ci
from herdsight.utils.config import REPO_ROOT, load_config
from herdsight.utils.env_report import env_report
from herdsight.utils.paths import storage_path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_baseline import LEVELS, best_threshold, load_preds  # noqa: E402

FARMS = ("Derval", "Jalogny", "Mauron")


def out_of_sample(
    keys: list[str],
    part_of: dict[str, str],
    preds: dict,
    truth: dict,
    grid: np.ndarray,
    edges: list[float],
) -> dict[str, np.ndarray]:
    """Counts and both methods' intervals for every key, each part fitted on the other parts."""
    n = len(keys)
    res = {"pred": np.zeros(n), "true": np.zeros(n)}
    for lv in LEVELS:
        for m in ("pooled", "band"):
            res[f"{m}_lo_{lv}"], res[f"{m}_hi_{lv}"] = np.zeros(n), np.zeros(n)
    fits = {}
    for part in sorted({part_of[k] for k in keys}):
        other = [k for k in keys if part_of[k] != part]
        own = np.array([i for i, k in enumerate(keys) if part_of[k] == part])
        thr = float(best_threshold(other, preds, truth, grid)[0])

        def count(ks: list[str], thr: float = thr) -> np.ndarray:
            return np.array([(preds[k][1] >= thr).sum() for k in ks], float)

        c_cal = count(other)
        t_cal = np.array([truth[k].shape[0] for k in other], float)
        pooled = fit_count_conformal(c_cal, t_cal, LEVELS)
        band = fit_band_conformal(c_cal, t_cal, edges, LEVELS)
        c_own = count([keys[i] for i in own])
        res["pred"][own] = c_own
        res["true"][own] = [truth[keys[i]].shape[0] for i in own]
        for lv in LEVELS:
            lo, hi = count_interval(c_own, pooled[lv])
            res[f"pooled_lo_{lv}"][own], res[f"pooled_hi_{lv}"][own] = lo, hi
            lo, hi = band_interval(c_own, band, lv)
            res[f"band_lo_{lv}"][own], res[f"band_hi_{lv}"][own] = lo, hi
        fits[part] = {
            "threshold": thr,
            "calibration_images": len(other),
            "pooled_q": {str(lv): q for lv, q in pooled.items()},
            "band_q": [
                {"n": b["n"], "q": {str(lv): q for lv, q in b["q"].items()}} for b in band["bands"]
            ],
        }
    res["fits"] = fits  # type: ignore[assignment]
    return res


def summarise(res: dict, groups: np.ndarray, edges: list[float], n_boot: int, seed: int) -> dict:
    t, p = res["true"], res["pred"]
    true_band = predicted_band(t, edges)  # the same edges, applied to the true count
    pred_band = predicted_band(p, edges)
    names = [f"{int(a)}-{int(b) - 1}" for a, b in zip(edges[:-1], edges[1:], strict=True)]
    names = [n if n != "0-0" else "0" for n in names] + [f"{int(edges[-1])}+"]

    def cell(mask: np.ndarray, method: str, lv: float) -> dict:
        if not mask.any():
            return {"n": 0}
        lo, hi = res[f"{method}_lo_{lv}"][mask], res[f"{method}_hi_{lv}"][mask]
        inside = ((t[mask] >= lo) & (t[mask] <= hi)).astype(float)
        rng = np.random.default_rng(seed)
        ci = group_bootstrap_ci(inside, groups[mask], np.mean, n_boot, rng)
        width = hi - lo
        return {
            "n": int(mask.sum()),
            "coverage": float(inside.mean()),
            "coverage_ci95": list(ci),
            "mean_width": float(np.mean(width)) if np.all(np.isfinite(width)) else None,
            "unbounded_share": float(np.mean(~np.isfinite(width))),
        }

    out: dict = {}
    for method in ("pooled", "band"):
        out[method] = {}
        for lv in LEVELS:
            everyone = np.ones(len(t), bool)
            out[method][str(lv)] = {
                "all": cell(everyone, method, lv),
                "by_true_band": {
                    nm: cell(true_band == b, method, lv) for b, nm in enumerate(names)
                },
                "by_predicted_band": {
                    nm: cell(pred_band == b, method, lv) for b, nm in enumerate(names)
                },
            }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="dfine_ref", help="cross-validation predictions")
    ap.add_argument("--lofo", default=None, help="leave-one-farm-out prefix, e.g. dfine_g6ref")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    chk = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/freeze.py"),
            "--phase",
            "phase2_prereg",
            "--check",
        ],
        capture_output=True,
        text=True,
    )
    print(chk.stdout.strip())
    if chk.returncode != 0:
        return 1
    ev, p2 = load_config("evaluation"), load_config("phase2")
    edges = [float(x) for x in p2.require("count_bands")]
    h11 = p2.require("hypotheses")["H11"]
    cap = float(p2.require("interval_width_cap_factor"))
    n_boot, seed = int(ev.require("n_boot")), int(ev.require("bootstrap_seed"))
    start, stop, step = ev.require("operating_threshold_grid")
    grid = np.round(np.arange(start, stop + 1e-9, step), 4)
    sp = json.loads((REPO_ROOT / "results" / "splits.json").read_text())["icaerus_v2"]
    folds = [f"fold{k}" for k in range(sp["n_folds"])]
    truth, fold_of, group_of, farm_of = {}, {}, {}, {}
    for s, problem in iter_images(default_root(), with_metadata=False):
        fold = sp["flights"].get(f"{s.farm}/{s.flight}")
        if problem is None and s.labelled and fold in folds:
            k = f"{s.farm}/{s.flight}/{s.path.name}"
            truth[k], fold_of[k], farm_of[k] = s.boxes_xyxy, fold, s.farm
            group_of[k] = FlightStats(s.farm, s.flight, 0, 0).group
    runs = storage_path("runs", "baselines")
    preds: dict = {}
    for f in folds:
        preds |= load_preds(runs / args.detector / f"{f}.jsonl")
    keys = sorted(k for k in truth if k in preds)
    groups = np.array([group_of[k] for k in keys])
    cv = out_of_sample(keys, fold_of, preds, truth, grid, edges)
    result: dict = {
        "note": (
            "Public-data results, not field results. Cross-validation intervals carry the small "
            "leakage of the Phase 1 method: the other folds' predictions, on which each fold's "
            "threshold and quantiles are fitted, come from models that trained on that fold."
        ),
        "detector": args.detector,
        "band_edges": edges,
        "n_images": len(keys),
        "images_missing_predictions": len(truth) - len(keys),
        "cross_validation": {
            "fits": cv["fits"],
            "coverage": summarise(cv, groups, edges, n_boot, seed),
        },
    }
    c90 = result["cross_validation"]["coverage"]
    top = f"{int(edges[-1])}+"
    band20, pooled20 = (
        c90["band"]["0.9"]["by_true_band"][top],
        c90["pooled"]["0.9"]["by_true_band"][top],
    )
    ratio = None
    if band20.get("mean_width") and pooled20.get("mean_width"):
        ratio = band20["mean_width"] / pooled20["mean_width"]
    result["H11"] = {
        "rule": (
            f"band-wise 90% intervals on {top} images: coverage at least "
            f"{h11['coverage_20plus_min']} and mean width at most {h11['width_ratio_max']} times "
            "the pooled method's width there (true-count band; cross-validation)"
        ),
        "coverage_20plus": band20.get("coverage"),
        "coverage_20plus_ci95": band20.get("coverage_ci95"),
        "width_ratio": ratio,
        "width_cap_factor": cap,
        "held": bool(
            band20.get("coverage") is not None
            and ratio is not None
            and band20["coverage"] >= float(h11["coverage_20plus_min"])
            and ratio <= float(h11["width_ratio_max"])
        ),
    }
    if args.lofo:
        lofo: dict = {}
        for farm in FARMS:
            for f in folds:
                part = load_preds(runs / f"{args.lofo}_{farm}" / f"{f}.jsonl")
                wrong = [k for k in part if farm_of.get(k, farm) != farm]
                if wrong:
                    raise SystemExit(f"{farm}: predictions for another farm, e.g. {wrong[0]}")
                lofo |= part
        lkeys = sorted(k for k in truth if k in lofo)
        lres = out_of_sample(lkeys, farm_of, lofo, truth, grid, edges)
        result["leave_one_farm_out"] = {
            "prefix": args.lofo,
            "n_images": len(lkeys),
            "fits": lres["fits"],
            "coverage": summarise(
                lres, np.array([group_of[k] for k in lkeys]), edges, n_boot, seed
            ),
        }
    result["environment"] = env_report()
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    brief = {
        "H11": result["H11"],
        "cv_90_all": {m: c90[m]["0.9"]["all"] for m in ("pooled", "band")},
    }
    if args.lofo:
        lc = result["leave_one_farm_out"]["coverage"]
        brief["lofo_90_all"] = {m: lc[m]["0.9"]["all"] for m in ("pooled", "band")}
    print(json.dumps(brief, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
