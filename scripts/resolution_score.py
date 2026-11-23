"""Score the resolution study (Phase 2, step G11).

Usage: python scripts/resolution_score.py --predictions <folder holding scale_* folders>

Each scale is compared with the same model's native-resolution predictions (runs/baselines/
dfine_g2), on the same images and at that model's fixed per-fold operating thresholds
(results/phase2/g2_comparison.json). Differences carry paired 95% intervals over farm-date groups.
Writes results/phase2/g11_resolution.json. Adult cattle on public data: the pixel figure is a lower
bound for calves, which are not in the data, and is not a field result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from herdsight.data.icaerus import default_root, iter_images
from herdsight.data.splits import FlightStats
from herdsight.evaluate.centre_match import box_centres, match_hungarian, short_side
from herdsight.evaluate.resolution import ScaleResult, smallest_passing
from herdsight.utils.config import REPO_ROOT, load_config
from herdsight.utils.paths import storage_path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_baseline import load_preds  # noqa: E402


def per_image(keys, preds, truth, thr, kappa, sim):  # noqa: ANN001, ANN201
    pred_n, ctp = [], []
    for k, t in zip(keys, thr, strict=True):
        boxes = preds[k][0][preds[k][1] >= t]
        pred_n.append(len(boxes))
        ctp.append(
            match_hungarian(
                box_centres(boxes),
                box_centres(truth[k]),
                np.maximum(short_side(truth[k]), 1.0),
                kappa,
                sim,
            ).tp
        )
    return np.array(pred_n, float), np.array(ctp, float)


def stats(pred_n, ctp, true_n, i):  # noqa: ANN001, ANN201
    empty = true_n[i] == 0
    return {
        "centre_recall": float(ctp[i].sum() / max(true_n[i].sum(), 1)),
        "count_mae": float(np.mean(np.abs(pred_n[i] - true_n[i]))),
        "empty_fp_per_image": float(pred_n[i][empty].mean()) if empty.any() else float("nan"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True)
    args = ap.parse_args(argv)
    cfg, ev = load_config("resolution_study"), load_config("evaluation")
    kappa, sim = float(ev.require("centre_kappa")), float(ev.require("centre_match_similarity"))
    scales = [float(x) for x in cfg.require("scales")]
    max_loss = float(cfg.require("max_recall_loss"))
    ref_px = float(cfg.require("reference_pixels"))
    comparison = json.loads((REPO_ROOT / "results/phase2/g2_comparison.json").read_text())
    thresholds = comparison["operating_thresholds"]["dfine_g2"]
    sp = json.loads((REPO_ROOT / "results" / "splits.json").read_text())["icaerus_v2"]
    folds = [f"fold{k}" for k in range(sp["n_folds"])]
    truth, fold_of, group_of = {}, {}, {}
    for s, problem in iter_images(default_root(), with_metadata=False):
        fold = sp["flights"].get(f"{s.farm}/{s.flight}")
        if problem is None and s.labelled and fold in folds:
            k = f"{s.farm}/{s.flight}/{s.path.name}"
            truth[k], fold_of[k] = s.boxes_xyxy, fold
            group_of[k] = FlightStats(s.farm, s.flight, 0, 0).group
    native: dict = {}
    degraded: dict[float, dict] = {x: {} for x in scales}
    for f in folds:
        native |= load_preds(storage_path("runs", "baselines", "dfine_g2") / f"{f}.jsonl")
        for x in scales:
            degraded[x] |= load_preds(args.predictions / f"scale_{x:.2f}" / f"{f}.jsonl")
    keys = sorted(truth)
    missing = {
        "native": sum(k not in native for k in keys),
        **{f"{x:.2f}": sum(k not in degraded[x] for k in keys) for x in scales},
    }
    if any(missing.values()):
        raise SystemExit(f"predictions missing: {missing}")
    thr = np.array([thresholds[fold_of[k]] for k in keys])
    true_n = np.array([truth[k].shape[0] for k in keys], float)
    groups = np.array([group_of[k] for k in keys])
    ug = np.unique(groups)
    members = [np.nonzero(groups == u)[0] for u in ug]
    rng = np.random.default_rng(int(ev.require("bootstrap_seed")))
    draws = [
        np.concatenate([members[g] for g in rng.integers(0, len(ug), len(ug))])
        for _ in range(int(ev.require("n_boot")))
    ]
    idx = np.arange(len(keys))
    base_n, base_tp = per_image(keys, native, truth, thr, kappa, sim)
    base = stats(base_n, base_tp, true_n, idx)
    table, passes = [], []
    for x in scales:
        pred_n, ctp = per_image(keys, degraded[x], truth, thr, kappa, sim)
        point = stats(pred_n, ctp, true_n, idx)
        diff = np.array(
            [
                stats(pred_n, ctp, true_n, i)["centre_recall"]
                - stats(base_n, base_tp, true_n, i)["centre_recall"]
                for i in draws
            ]
        )
        change = point["centre_recall"] - base["centre_recall"]
        low, high = np.percentile(diff, [2.5, 97.5])
        passes.append(ScaleResult(x, change, float(low)))
        table.append(
            {
                "scale": x,
                "reference_animal_px": ref_px * x,
                **point,
                "recall_change": change,
                "recall_change_ci95": [float(low), float(high)],
            }
        )
    by_point, by_interval = smallest_passing(passes, max_loss)
    result = {
        "note": "Adult cattle, public data, simulated degradation. A lower bound for calves, "
        "which are not in the data. Not a field result.",
        "model": "dfine_g2",
        "n_images": len(keys),
        "n_groups": len(ug),
        "native": base,
        "max_recall_loss": max_loss,
        "by_scale": table,
        "smallest_passing_scale": by_point,
        "smallest_passing_px": None if by_point is None else ref_px * by_point,
        "smallest_passing_scale_interval_rule": by_interval,
        "smallest_passing_px_interval_rule": None if by_interval is None else ref_px * by_interval,
    }
    out = REPO_ROOT / "results" / "phase2" / "g11_resolution.json"
    out.write_bytes((json.dumps(result, indent=1) + "\n").encode("utf-8"))
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
