"""Score a baseline's out-of-fold predictions by grouped cross-validation (D-26).

Usage: python scripts/score_baseline.py --detector zeroshot|dfine

Reads predictions from $HERDSIGHT_STORAGE/runs/baselines/<detector>/fold<k>.jsonl and the labels.
Writes results/baselines/<detector>_scores.json.

For each fold k (fixed in docs/hypotheses.md before any run):
  1. the operating score threshold minimises per-image count MAE on the OTHER folds' predictions;
  2. conformal count intervals are fitted on the other folds' counts at that threshold;
  3. fold k is scored with both fixed.
Results are pooled over all folds; 95% intervals are bootstrap over farm-date groups.
When the detector is D-FINE and zero-shot predictions exist, D-FINE is also scored on the
zero-shot image sample, so H1 compares the two on identical images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from herdsight.data.icaerus import default_root, iter_images
from herdsight.data.splits import FlightStats
from herdsight.evaluate.centre_match import (
    ImageDetections,
    average_precision,
    box_centres,
    short_side,
)
from herdsight.evaluate.counts import count_errors, count_interval, fit_count_conformal
from herdsight.evaluate.stats import group_bootstrap_ci
from herdsight.evaluate.upright_ap import ImageBoxes, match_iou, upright_ap
from herdsight.utils.config import REPO_ROOT, Config, load_config
from herdsight.utils.env_report import env_report
from herdsight.utils.paths import storage_path

BANDS = (0, 1, 5, 20, math.inf)
LEVELS = (0.68, 0.90, 0.95)


def load_preds(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line cut short by an interruption
        out[r["image"]] = (
            np.asarray(r["boxes"], float).reshape(-1, 4),
            np.asarray(r["scores"], float),
        )
    return out


def best_threshold(
    keys: list[str], preds: dict, truth: dict, grid: np.ndarray
) -> tuple[float, float]:
    true_n = np.array([truth[k].shape[0] for k in keys], float)
    maes = [
        float(np.mean(np.abs(np.array([(preds[k][1] >= t).sum() for k in keys]) - true_n)))
        for t in grid
    ]
    i = int(np.argmin(maes))
    return float(grid[i]), maes[i]


def cross_fit(
    keys: list[str], fold_of: dict[str, str], preds: dict, truth: dict, grid: np.ndarray
) -> dict[str, dict]:
    """Per fold: threshold and conformal quantiles from the other folds only."""
    out = {}
    for k in sorted({fold_of[x] for x in keys}):
        other = [x for x in keys if fold_of[x] != k]
        thr, mae = best_threshold(other, preds, truth, grid)
        pred_c = np.array([(preds[x][1] >= thr).sum() for x in other], float)
        true_c = np.array([truth[x].shape[0] for x in other], float)
        out[k] = {
            "threshold": thr,
            "fit_mae": mae,
            "fit_images": len(other),
            "q": fit_count_conformal(pred_c, true_c, LEVELS),
        }
    return out


def evaluate(
    keys: list[str],
    preds: dict,
    truth: dict,
    fold_of: dict[str, str],
    group_of: dict[str, str],
    alt_of: dict[str, float | None],
    fits: dict[str, dict],
    ev: Config,
) -> dict[str, object]:
    rng = np.random.default_rng(int(ev.require("bootstrap_seed")))
    n_boot, n_boot_ap = int(ev.require("n_boot")), int(ev.require("n_boot_ap"))
    iou_t = float(ev.require("iou_match_threshold"))
    groups = np.array([group_of[k] for k in keys])
    thr = np.array([fits[fold_of[k]]["threshold"] for k in keys])
    true_n = np.array([truth[k].shape[0] for k in keys], float)
    kept = [preds[k][1] >= t for k, t in zip(keys, thr, strict=True)]
    pred_n = np.array([m.sum() for m in kept], float)
    tp = np.array(
        [match_iou(preds[k][0][m], truth[k], iou_t) for k, m in zip(keys, kept, strict=True)],
        float,
    )
    fp = pred_n - tp
    idx = np.arange(len(keys))

    def ci(stat: Callable[[np.ndarray], float], sub: np.ndarray = idx) -> list[float] | None:
        if len(set(groups[sub])) < 2:
            return None
        return list(group_bootstrap_ci(sub, groups[sub], stat, n_boot, rng))

    out: dict[str, object] = {
        "n_images": len(keys),
        "n_groups": len(set(groups)),
        "n_true": int(true_n.sum()),
    }
    out["counts"] = count_errors(pred_n, true_n)
    out["counts"]["mae_ci95"] = ci(lambda i: float(np.mean(np.abs(pred_n[i] - true_n[i]))))
    out["recall"] = float(tp.sum() / max(true_n.sum(), 1))
    out["recall_ci95"] = ci(lambda i: float(tp[i].sum() / max(true_n[i].sum(), 1)))
    out["precision"] = float(tp.sum() / max(pred_n.sum(), 1))
    out["precision_ci95"] = ci(lambda i: float(tp[i].sum() / max(pred_n[i].sum(), 1)))

    boxes = [ImageBoxes.make(*preds[k], truth[k]) for k in keys]
    out["upright_ap"] = upright_ap(boxes)

    def ap50(i: np.ndarray) -> float:
        return upright_ap([boxes[j] for j in i])["ap50"]

    out["upright_ap"]["ap50_ci95"] = (
        list(group_bootstrap_ci(idx, groups, ap50, n_boot_ap, rng))
        if len(set(groups)) > 1
        else None
    )

    rel = []  # H2, and H5 (the empty band)
    for lo, hi in zip(BANDS[:-1], BANDS[1:], strict=True):
        m = np.nonzero((true_n >= lo) & (true_n < hi))[0]
        row: dict[str, object] = {"band": [lo, hi if math.isfinite(hi) else None], "n": int(m.size)}
        if m.size and lo > 0:
            row["relative_mae"] = float(np.mean(np.abs(pred_n[m] - true_n[m]) / true_n[m]))
            row["relative_mae_ci95"] = ci(
                lambda i: float(np.mean(np.abs(pred_n[i] - true_n[i]) / true_n[i])), m
            )
        if m.size and lo == 0:
            row["false_positives_per_image"] = float(fp[m].mean())
            row["false_positives_per_image_ci95"] = ci(lambda i: float(fp[i].mean()), m)
        rel.append(row)
    out["by_band"] = rel

    intervals = {}  # H3: each image uses its own fold's quantile
    for lv in LEVELS:
        lo = np.empty(len(keys))
        hi = np.empty(len(keys))
        for j, k in enumerate(keys):
            a, b = count_interval([pred_n[j]], fits[fold_of[k]]["q"][lv])
            lo[j], hi[j] = a[0], b[0]
        inside = (true_n >= lo) & (true_n <= hi)
        bands = []
        for lo_b, hi_b in zip(BANDS[:-1], BANDS[1:], strict=True):
            m = (true_n >= lo_b) & (true_n < hi_b)
            bands.append(
                {
                    "band": [lo_b, hi_b if math.isfinite(hi_b) else None],
                    "n": int(m.sum()),
                    "coverage": float(inside[m].mean()) if m.any() else None,
                }
            )
        intervals[f"{lv:.2f}"] = {
            "coverage": float(inside.mean()),
            "coverage_ci95": ci(lambda i, inside=inside: float(inside[i].mean())),
            "mean_width": float(np.mean(hi - lo)),
            "coverage_by_band": bands,
        }
    out["intervals"] = intervals

    alts = np.array([alt_of.get(k) or np.nan for k in keys], float)  # H4
    by_alt = {}
    for a in sorted({round(x / 10) * 10 for x in alts[np.isfinite(alts)]}):
        m = np.nonzero(np.isfinite(alts) & (np.abs(alts - a) < 5))[0]
        by_alt[f"{a:.0f}m"] = {
            "n_images": int(m.size),
            "n_true": int(true_n[m].sum()),
            "recall": float(tp[m].sum() / max(true_n[m].sum(), 1)),
            "recall_ci95": ci(lambda i: float(tp[i].sum() / max(true_n[i].sum(), 1)), m),
        }
    out["recall_by_altitude"] = by_alt

    imgs = [
        ImageDetections.make(
            box_centres(preds[k][0]),
            preds[k][1],
            box_centres(truth[k]),
            np.maximum(short_side(truth[k]), 1.0),
        )
        for k in keys
    ]
    kappa = ev.get("centre_kappa")
    if kappa is None:
        out["centre_ap"] = {
            "status": "pending: centre_kappa unresolved (owner re-annotation)",
            "sensitivity_only": {
                str(kp): average_precision(imgs, kp)["ap_mean"]
                for kp in ev.require("centre_kappa_sensitivity")
            },
        }
    else:
        out["centre_ap"] = average_precision(imgs, float(kappa))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True, help="zeroshot or dfine (a runs subfolder)")
    ap.add_argument("--out", type=Path, default=None, help="output file (default: results/)")
    args = ap.parse_args(argv)
    ev = load_config("evaluation")
    splits_path = REPO_ROOT / "results" / "splits.json"
    sp = json.loads(splits_path.read_text())["icaerus_v2"]
    folds = [f"fold{k}" for k in range(sp["n_folds"])]
    runs = storage_path("runs", "baselines")
    preds: dict = {}
    for f in folds:
        preds |= load_preds(runs / args.detector / f"{f}.jsonl")
    truth, fold_of, group_of = {}, {}, {}
    for s, problem in iter_images(default_root(), with_metadata=False):
        fold = sp["flights"].get(f"{s.farm}/{s.flight}")
        if problem is None and s.labelled and fold in folds:
            k = f"{s.farm}/{s.flight}/{s.path.name}"
            truth[k], fold_of[k] = s.boxes_xyxy, fold
            group_of[k] = FlightStats(s.farm, s.flight, 0, 0).group
    alt_of: dict[str, float | None] = {}
    with (REPO_ROOT / "results" / "data" / "icaerus_images.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            v = r["rel_alt_m"]
            alt_of[f"{r['farm']}/{r['flight']}/{r['image']}"] = float(v) if v else None

    keys = sorted(k for k in preds if k in truth)
    missing = sorted(set(truth) - set(preds))
    start, stop, step = ev.require("operating_threshold_grid")
    grid = np.round(np.arange(start, stop + 1e-9, step), 4)
    fits = cross_fit(keys, fold_of, preds, truth, grid)
    result = {
        "detector": args.detector,
        "design": "grouped 3-fold cross-validation over farm-date groups (D-26); per fold, "
        "threshold (min count MAE) and conformal intervals fitted on the other folds",
        "splits_sha256": hashlib.sha256(splits_path.read_bytes()).hexdigest(),
        "evaluation_config_hash": ev.hash(),
        "n_images_scored": len(keys),
        "n_images_without_predictions": len(missing),
        "per_fold_fit": {
            k: {**v, "q": {f"{lv:.2f}": q for lv, q in v["q"].items()}} for k, v in fits.items()
        },
        "out_of_fold": evaluate(keys, preds, truth, fold_of, group_of, alt_of, fits, ev),
        "environment": env_report(),
    }
    if args.detector == "dfine":
        zs: dict = {}
        for f in folds:
            zs |= load_preds(runs / "zeroshot" / f"{f}.jsonl")
        sample = sorted(k for k in zs if k in preds and k in truth)
        if sample:
            result["out_of_fold_on_zeroshot_sample"] = evaluate(
                sample, preds, truth, fold_of, group_of, alt_of, fits, ev
            )
    dest = args.out or REPO_ROOT / "results" / "baselines" / f"{args.detector}_scores.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    o = result["out_of_fold"]
    print(
        json.dumps(
            {
                "images": len(keys),
                "mae": o["counts"]["mae"],
                "mae_ci95": o["counts"]["mae_ci95"],
                "ap50": o["upright_ap"]["ap50"],
                "recall": o["recall"],
                "precision": o["precision"],
                "coverage_90": o["intervals"]["0.90"]["coverage"],
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
