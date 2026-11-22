"""Phase 2: paired comparison of a candidate detector with the reference (selection rule, H7, H8).

Usage: python scripts/phase2_compare.py --candidate dfine_g2 --reference dfine_ref --out <json>

Both are folders of out-of-fold full-image predictions under $HERDSIGHT_STORAGE/runs/baselines/
(scripts/run_baseline.py --name). Each detector gets its own per-fold operating threshold, fitted
on the other folds (count MAE minimiser, scripts/score_baseline.py). Measured on every labelled
image of the cross-validation farms:

* centre AP (mean over similarity thresholds) and count MAE at the operating point;
* centre-matched recall per farm-date group, and the worst group's;
* Derval centre-matched recall and pooled centre-matched recall (H7);
* false detections per empty image (H8).

Differences (candidate - reference) carry 95% intervals from one set of farm-date group
resamples shared by both detectors (paired). The selection rule and hypothesis thresholds are
read from configs/phase2.yaml (frozen; the freeze is checked first).
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
from herdsight.evaluate.centre_match import (
    ImageDetections,
    ap_mean_subset,
    box_centres,
    match_hungarian,
    match_table,
    short_side,
)
from herdsight.utils.config import REPO_ROOT, load_config
from herdsight.utils.env_report import env_report
from herdsight.utils.paths import storage_path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_baseline import cross_fit, load_preds  # noqa: E402


def measures(keys, preds, truth, fold_of, fits, kappa, sim):  # noqa: ANN001, ANN201
    thr = np.array([fits[fold_of[k]]["threshold"] for k in keys])
    true_n = np.array([truth[k].shape[0] for k in keys], float)
    kept = [preds[k][1] >= t for k, t in zip(keys, thr, strict=True)]
    pred_n = np.array([m.sum() for m in kept], float)
    ctp = np.array(
        [
            match_hungarian(
                box_centres(preds[k][0][m]),
                box_centres(truth[k]),
                np.maximum(short_side(truth[k]), 1.0),
                kappa,
                sim,
            ).tp
            for k, m in zip(keys, kept, strict=True)
        ],
        float,
    )
    imgs = [
        ImageDetections.make(
            box_centres(preds[k][0]),
            preds[k][1],
            box_centres(truth[k]),
            np.maximum(short_side(truth[k]), 1.0),
        )
        for k in keys
    ]
    return {
        "true_n": true_n,
        "pred_n": pred_n,
        "ctp": ctp,
        "imgs": imgs,
        "table": match_table(imgs, kappa),
    }


def stats(m: dict, i: np.ndarray, groups: np.ndarray, derval: np.ndarray, empty: np.ndarray):  # noqa: ANN201
    t, p, c = m["true_n"][i], m["pred_n"][i], m["ctp"][i]
    g = groups[i]
    per_group = {
        u: float(c[g == u].sum() / max(t[g == u].sum(), 1)) for u in np.unique(g) if t[g == u].sum()
    }
    dv, em = derval[i], empty[i]
    return {
        "centre_ap": ap_mean_subset(m["imgs"], m["table"], i),
        "count_mae": float(np.mean(np.abs(p - t))),
        "centre_recall": float(c.sum() / max(t.sum(), 1)),
        "worst_group_recall": min(per_group.values()) if per_group else float("nan"),
        "derval_recall": float(c[dv].sum() / max(t[dv].sum(), 1)) if dv.any() else float("nan"),
        "empty_fp_per_image": float(p[em].mean()) if em.any() else float("nan"),
        "per_group_recall": per_group,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    chk = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "freeze.py"),
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
    kappa, sim = float(ev.require("centre_kappa")), float(ev.require("centre_match_similarity"))
    hyp = p2.require("hypotheses")
    close = float(p2.require("close_call_delta"))
    sp = json.loads((REPO_ROOT / "results" / "splits.json").read_text())["icaerus_v2"]
    folds = [f"fold{k}" for k in range(sp["n_folds"])]
    truth, fold_of, group_of = {}, {}, {}
    for s, problem in iter_images(default_root(), with_metadata=False):
        fold = sp["flights"].get(f"{s.farm}/{s.flight}")
        if problem is None and s.labelled and fold in folds:
            k = f"{s.farm}/{s.flight}/{s.path.name}"
            truth[k], fold_of[k] = s.boxes_xyxy, fold
            group_of[k] = FlightStats(s.farm, s.flight, 0, 0).group
    runs = storage_path("runs", "baselines")
    preds = {}
    for name in (args.candidate, args.reference):
        preds[name] = {}
        for f in folds:
            preds[name] |= load_preds(runs / name / f"{f}.jsonl")
    keys = sorted(k for k in truth if all(k in preds[n] for n in preds))
    missing = len(truth) - len(keys)
    start, stop, step = ev.require("operating_threshold_grid")
    grid = np.round(np.arange(start, stop + 1e-9, step), 4)
    groups = np.array([group_of[k] for k in keys])
    derval = np.array([k.startswith("Derval/") for k in keys])
    empty = np.array([truth[k].shape[0] == 0 for k in keys])
    m, fits, point = {}, {}, {}
    idx = np.arange(len(keys))
    for name in preds:
        fits[name] = cross_fit(keys, fold_of, preds[name], truth, grid)
        m[name] = measures(keys, preds[name], truth, fold_of, fits[name], kappa, sim)
        point[name] = stats(m[name], idx, groups, derval, empty)
    rng = np.random.default_rng(int(ev.require("bootstrap_seed")))
    ug = np.unique(groups)
    members = [np.nonzero(groups == u)[0] for u in ug]
    names = (
        "centre_ap",
        "count_mae",
        "centre_recall",
        "worst_group_recall",
        "derval_recall",
        "empty_fp_per_image",
    )
    diffs = {n: [] for n in names}
    for _ in range(int(ev.require("n_boot_ap"))):
        i = np.concatenate([members[g] for g in rng.integers(0, len(ug), len(ug))])
        a = stats(m[args.candidate], i, groups, derval, empty)
        b = stats(m[args.reference], i, groups, derval, empty)
        for n in names:
            diffs[n].append(a[n] - b[n])
    paired = {}
    for n in names:
        v = np.array(diffs[n])
        v = v[np.isfinite(v)]
        paired[n] = {
            "difference": point[args.candidate][n] - point[args.reference][n],
            "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
        }
    ap_d, mae_d = paired["centre_ap"], paired["count_mae"]
    better = ap_d["ci95"][0] > 0 and mae_d["ci95"][1] < 0
    no_loss_worst = paired["worst_group_recall"]["difference"] >= 0
    close_call = abs(ap_d["difference"]) < close or ap_d["ci95"][0] <= 0 <= ap_d["ci95"][1]
    close_call |= mae_d["ci95"][0] <= 0 <= mae_d["ci95"][1]
    h7 = hyp["H7"]
    h8 = hyp["H8"]
    ref_fp = point[args.reference]["empty_fp_per_image"]
    result = {
        "candidate": args.candidate,
        "reference": args.reference,
        "n_images": len(keys),
        "n_groups": len(ug),
        "images_missing_predictions": missing,
        "operating_thresholds": {n: {f: v["threshold"] for f, v in fits[n].items()} for n in fits},
        "point": {n: {k: v for k, v in point[n].items() if k != "per_group_recall"} for n in point},
        "per_group_recall": {n: point[n]["per_group_recall"] for n in point},
        "paired": paired,
        "selection_rule": {
            "better_on_centre_ap_and_count_mae": bool(better),
            "no_loss_on_worst_group_recall": bool(no_loss_worst),
            "replaces_reference": bool(better and no_loss_worst),
            "close_call_repeats_needed": bool(close_call),
        },
        "H7": {
            "derval_recall_gain": paired["derval_recall"]["difference"],
            "pooled_centre_recall_change": paired["centre_recall"]["difference"],
            "holds": bool(
                paired["derval_recall"]["difference"] >= h7["derval_recall_gain_min"]
                and paired["centre_recall"]["difference"] >= -h7["pooled_centre_recall_loss_max"]
            ),
        },
        "H8": {
            "empty_fp_ratio": point[args.candidate]["empty_fp_per_image"] / ref_fp
            if ref_fp
            else None,
            "holds": bool(
                ref_fp
                and point[args.candidate]["empty_fp_per_image"] / ref_fp <= h8["empty_fp_ratio_max"]
            ),
        },
        "environment": env_report(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    for n in names:
        print(
            f"{n:20s} {point[args.candidate][n]:.4f} vs {point[args.reference][n]:.4f}  "
            f"diff {paired[n]['difference']:+.4f} {np.round(paired[n]['ci95'], 4)}"
        )
    print("selection rule:", result["selection_rule"])
    print("H7:", result["H7"], "H8:", result["H8"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
