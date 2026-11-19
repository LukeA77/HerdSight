"""Score onboard herd detection (S7, D-28) from out-of-fold predictions.

Usage: python scripts/score_herds.py [--detector dfine_onboard]

Reads $HERDSIGHT_STORAGE/runs/baselines/<detector>/fold<k>.jsonl (boxes in full-image pixels) and
the labels of every cross-validation image. For each fold k the score threshold is the per-image
count-MAE minimiser on the other folds (the same rule as scripts/score_baseline.py), so no fold
is scored with a threshold chosen on itself. Images without a relative altitude or focal length
cannot be put in metres and are skipped (reported).

Writes results/baselines/<detector>_herds.json. Latency is NOT measured here: it is reported as
not measured until the model runs on a Raspberry Pi 5 (D-10, D-29).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from herdsight.data.icaerus import default_root, iter_images
from herdsight.data.splits import FlightStats
from herdsight.evaluate.herds import score_image
from herdsight.evaluate.stats import group_bootstrap_ci
from herdsight.geo.nadir import focal_px_from_35mm
from herdsight.utils.config import REPO_ROOT, load_config
from herdsight.utils.env_report import env_report
from herdsight.utils.paths import storage_path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_baseline import cross_fit, load_preds  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="dfine_onboard")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    ev = load_config("evaluation")
    link_m = float(ev.require("herd_link_m"))
    min_animals = int(ev.require("herd_min_animals"))
    splits_path = REPO_ROOT / "results" / "splits.json"
    sp = json.loads(splits_path.read_text())["icaerus_v2"]
    folds = [f"fold{k}" for k in range(sp["n_folds"])]
    runs = storage_path("runs", "baselines", args.detector)
    preds: dict = {}
    for f in folds:
        preds |= load_preds(runs / f"{f}.jsonl")
    if not preds:
        print(f"no predictions in {runs}", file=sys.stderr)
        return 1

    gsd: dict[str, float] = {}
    with (REPO_ROOT / "results" / "data" / "icaerus_images.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["rel_alt_m"] and r["focal_35mm"]:
                k = f"{r['farm']}/{r['flight']}/{r['image']}"
                gsd[k] = (float(r["rel_alt_m"]), float(r["focal_35mm"]))  # type: ignore[assignment]
    truth, fold_of, group_of, gsd_m = {}, {}, {}, {}
    for s, problem in iter_images(default_root(), with_metadata=False):
        fold = sp["flights"].get(f"{s.farm}/{s.flight}")
        k = f"{s.farm}/{s.flight}/{s.path.name}"
        if problem is None and s.labelled and fold in folds:
            truth[k], fold_of[k] = s.boxes_xyxy, fold
            group_of[k] = FlightStats(s.farm, s.flight, 0, 0).group
            if k in gsd:
                alt, f35 = gsd[k]  # type: ignore[misc]
                gsd_m[k] = alt / focal_px_from_35mm(f35, s.width, s.height)

    keys = sorted(k for k in preds if k in truth)
    start, stop, step = ev.require("operating_threshold_grid")
    grid = np.round(np.arange(start, stop + 1e-9, step), 4)
    fits = cross_fit(keys, fold_of, preds, truth, grid)
    scored = [k for k in keys if k in gsd_m]
    rows = []
    for k in scored:
        boxes, scores = preds[k]
        kept = boxes[scores >= fits[fold_of[k]]["threshold"]]
        rows.append(score_image(truth[k], kept, gsd_m[k], link_m, min_animals))
    th = np.array([r.true_herds for r in rows], float)
    fo = np.array([r.found for r in rows], float)
    gr = np.array([r.grouped for r in rows], float)
    ft = np.array([r.false_triggers for r in rows], float)
    groups = np.array([group_of[k] for k in scored])
    rng = np.random.default_rng(int(ev.require("bootstrap_seed")))
    n_boot = int(ev.require("n_boot"))
    idx = np.arange(len(scored))

    def ci(stat):  # noqa: ANN001, ANN202
        return list(group_bootstrap_ci(idx, groups, stat, n_boot, rng))

    no_herd = th == 0
    result = {
        "detector": args.detector,
        "definition": f"herd = >= {min_animals} animals single-linked within {link_m} m (D-28)",
        "design": "grouped 3-fold cross-validation (D-26); threshold per fold from the other folds",
        "splits_sha256": hashlib.sha256(splits_path.read_bytes()).hexdigest(),
        "evaluation_config_hash": ev.hash(),
        "n_images_with_predictions": len(keys),
        "n_images_scored": len(scored),
        "n_images_skipped_no_scale": len(keys) - len(scored),
        "n_groups": len(set(groups)),
        "true_herds": int(th.sum()),
        "per_fold_threshold": {k: v["threshold"] for k, v in fits.items()},
        "herd_recall_found": float(fo.sum() / max(th.sum(), 1)),
        "herd_recall_found_ci95": ci(lambda i: float(fo[i].sum() / max(th[i].sum(), 1))),
        "herd_recall_grouped": float(gr.sum() / max(th.sum(), 1)),
        "herd_recall_grouped_ci95": ci(lambda i: float(gr[i].sum() / max(th[i].sum(), 1))),
        "false_triggers_per_image": float(ft.mean()) if len(ft) else None,
        "false_triggers_per_image_ci95": ci(lambda i: float(ft[i].mean())),
        "images_without_a_true_herd": int(no_herd.sum()),
        "false_trigger_share_of_herd_free_images": (
            float((ft[no_herd] > 0).mean()) if no_herd.any() else None
        ),
        "latency": "not measured: requires a Raspberry Pi 5 (D-10, D-29)",
        "environment": env_report(),
    }
    dest = args.out or REPO_ROOT / "results" / "baselines" / f"{args.detector}_herds.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "environment"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
