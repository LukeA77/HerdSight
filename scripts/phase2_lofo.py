"""Phase 2 step G6: how the reference recipe does on a farm its model never saw (D-33).

Usage: python scripts/phase2_lofo.py --prefix dfine_g6ref --reference dfine_ref --out <json>

Each farm's images are predicted by a model trained only on the other two farms
(runs/baselines/<prefix>_<farm>/). Its operating threshold is the count-error minimiser on the
other two farms' leave-one-farm-out predictions, so nothing from the held-out farm sets it. The
reference column is the same recipe under the official cross-validation, where a farm's other
survey days can be in training (Derval has only one day, so it is unseen in both). The drop from
the reference to leave-one-farm-out shows how much the official figure depends on knowing the farm.
Differences carry 95% intervals from paired resampling of farm-date groups. Public-data results,
not field results.
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
from herdsight.utils.config import REPO_ROOT, load_config
from herdsight.utils.env_report import env_report
from herdsight.utils.paths import storage_path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_compare import measures, stats  # noqa: E402
from score_baseline import best_threshold, cross_fit, load_preds  # noqa: E402

FARMS = ("Derval", "Jalogny", "Mauron")
REPORTED = ("centre_ap", "count_mae", "centre_recall", "empty_fp_per_image")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="dfine_g6ref")
    ap.add_argument("--reference", default="dfine_ref")
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
    ev = load_config("evaluation")
    kappa, sim = float(ev.require("centre_kappa")), float(ev.require("centre_match_similarity"))
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
    lofo: dict = {}
    for farm in FARMS:
        for f in folds:
            part = load_preds(runs / f"{args.prefix}_{farm}" / f"{f}.jsonl")
            wrong = [k for k in part if farm_of.get(k, farm) != farm]
            if wrong:
                raise SystemExit(f"{farm}: predictions for another farm, e.g. {wrong[0]}")
            lofo |= part
    reference: dict = {}
    for f in folds:
        reference |= load_preds(runs / args.reference / f"{f}.jsonl")
    keys = sorted(k for k in truth if k in lofo and k in reference)
    missing = len(truth) - len(keys)
    start, stop, step = ev.require("operating_threshold_grid")
    grid = np.round(np.arange(start, stop + 1e-9, step), 4)
    fits_lofo = {}
    for farm in FARMS:
        other = [k for k in keys if farm_of[k] != farm]
        fits_lofo[farm] = {"threshold": float(best_threshold(other, lofo, truth, grid)[0])}
    m_lofo = measures(keys, lofo, truth, farm_of, fits_lofo, kappa, sim)
    fits_ref = cross_fit(keys, fold_of, reference, truth, grid)
    m_ref = measures(keys, reference, truth, fold_of, fits_ref, kappa, sim)
    groups = np.array([group_of[k] for k in keys])
    derval = np.array([farm_of[k] == "Derval" for k in keys])
    empty = np.array([truth[k].shape[0] == 0 for k in keys])
    farms = np.array([farm_of[k] for k in keys])

    def row(m: dict, i: np.ndarray) -> dict:
        s = stats(m, i, groups, derval, empty)
        return {n: s[n] for n in REPORTED}

    idx = np.arange(len(keys))
    by_farm = {
        farm: {
            "images": int((farms == farm).sum()),
            "animals": int(sum(truth[k].shape[0] for k in keys if farm_of[k] == farm)),
            "farm_date_groups": int(len(np.unique(groups[farms == farm]))),
            "leave_one_farm_out": row(m_lofo, np.flatnonzero(farms == farm)),
            "official_cross_validation": row(m_ref, np.flatnonzero(farms == farm)),
        }
        for farm in FARMS
    }
    rng = np.random.default_rng(int(ev.require("bootstrap_seed")))
    ug = np.unique(groups)
    members = [np.nonzero(groups == u)[0] for u in ug]
    diffs: dict[str, list[float]] = {n: [] for n in REPORTED}
    for _ in range(int(ev.require("n_boot_ap"))):
        i = np.concatenate([members[g] for g in rng.integers(0, len(ug), len(ug))])
        a, b = row(m_lofo, i), row(m_ref, i)
        for n in REPORTED:
            diffs[n].append(a[n] - b[n])
    pooled_lofo, pooled_ref = row(m_lofo, idx), row(m_ref, idx)
    paired = {}
    for n in REPORTED:
        v = np.array(diffs[n])
        v = v[np.isfinite(v)]
        paired[n] = {
            "difference": pooled_lofo[n] - pooled_ref[n],
            "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
        }
    result = {
        "note": (
            "Public-data results, not field results. Leave-one-farm-out models use the reference "
            "recipe and were trained on a different GPU type from the reference (D-35)."
        ),
        "n_images": len(keys),
        "images_missing_predictions": missing,
        "operating_thresholds_lofo": {f: v["threshold"] for f, v in fits_lofo.items()},
        "operating_thresholds_reference": {f: v["threshold"] for f, v in fits_ref.items()},
        "by_farm": by_farm,
        "pooled": {"leave_one_farm_out": pooled_lofo, "official_cross_validation": pooled_ref},
        "paired_lofo_minus_official": paired,
        "environment": env_report(),
    }
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("by_farm", "pooled", "paired_lofo_minus_official")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
