"""Phase 3 step 3: candidate pairs, their evidence and the out-of-fold same-animal model.

Usage: python scripts/phase3_pairs.py

For the 20 owner-verified flights (configs/phase3.yaml, frozen), with two sources of detections:
  labels   the labelled boxes (the linking alone, as in the error split);
  dfine    the Phase 1 detector's out-of-fold detections at its per-fold operating threshold
           (results/phase3/dfine_s3_v20.json).
Candidate pairs between overlapping stills and their features come from
herdsight.count.evidence. Each pair is labelled from the owner's review
(herdsight.count.truth): 1 if both boxes belong to one verified animal, 0 if to different ones,
unknown inside a group the review left with two boxes from one image. A detection is tied to a
labelled box by centre matching; a pair with one unmatched detection (a false detection) is 0,
and a pair of two unmatched detections is unknown.

Everything learned is fitted on two folds and applied to the third: the grazing allowance (growth
of the squared position difference with time, from linked label pairs) and a logistic model per
source and feature set ("movement", "fixed"; H24 compares them in the grouping step). The
out-of-fold area under the ROC curve is reported with a bootstrap interval over flights.

Writes the pair tables with out-of-fold log odds to
$HERDSIGHT_STORAGE/processed/phase3/pairs_<source>.npz (with each still's detections, read by
the grouping and appearance steps) and the
summary to results/phase3/pair_model.json.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from herdsight.count.evidence import (
    FEATURES,
    MotionModel,
    Overlap,
    Still,
    auc,
    candidates,
    features,
    fit_logistic,
    match_to_labels,
)
from herdsight.count.truth import verified_identities
from herdsight.data.icaerus import default_root, iter_images
from herdsight.utils.config import REPO_ROOT, load_config
from herdsight.utils.env_report import env_report
from herdsight.utils.paths import storage_path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from link_animals import centres  # noqa: E402
from score_baseline import load_preds  # noqa: E402
from score_s3 import flight_geometry, ground  # noqa: E402

SOURCES = ("labels", "dfine")
KINDS = ("movement", "fixed")
CAND_FIELDS = ("i", "a", "j", "b", "d_m", "dt_s", "geotag", "fixed_gate_m", "size", "low_score")


def box_len(b: np.ndarray) -> np.ndarray:
    return np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]) if len(b) else np.zeros(0)


def main() -> int:
    p3 = load_config("phase3")
    boot = p3.require("bootstrap")
    truth = json.loads((REPO_ROOT / p3.require("truth")).read_text())["flights"]
    keys = sorted(k for k, v in truth.items() if v["complete"])
    sp = json.loads((REPO_ROOT / p3.require("folds")).read_text())["icaerus_v2"]
    fold_of = {k: sp["flights"][k] for k in keys}
    folds = sorted(set(fold_of.values()))
    det_s3 = json.loads((REPO_ROOT / "results" / "phase3" / "dfine_s3_v20.json").read_text())
    thresholds = det_s3["per_fold_threshold"]

    meta, sizes = {}, {}
    with (REPO_ROOT / "results" / "data" / "icaerus_images.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            meta[(r["farm"], r["flight"], r["image"])] = r
    wanted = {tuple(k.split("/")) for k in keys}
    for s, problem in iter_images(default_root(), with_metadata=False):
        if problem is None and (s.farm, s.flight) in wanted:
            sizes[(s.farm, s.flight, s.path.name)] = (s.width, s.height)
    preds: dict = {}
    for f in folds:
        preds |= load_preds(storage_path("runs", "baselines", "dfine") / f"{f}.jsonl")

    links_dir = storage_path("processed", "s3_links")
    raw: dict[str, list] = {s: [] for s in SOURCES}
    boxes_of: dict[str, dict] = {s: {} for s in SOURCES}  # detections per still, for crops
    for key in keys:
        farm, flight = key.split("/")
        det = json.loads((links_dir / f"{farm}__{flight}.json").read_text(encoding="utf-8"))
        names = det["images"]
        g = flight_geometry(farm, flight, names, meta, sizes)
        times = [
            datetime.strptime(meta[(farm, flight, n)]["time"], "%Y:%m:%d %H:%M:%S").timestamp()
            for n in names
        ]
        overlaps = [
            Overlap(p["i"], p["j"], None if p["H_ab"] is None else np.asarray(p["H_ab"]))
            for p in det["pairs"]
        ]
        rev = verified_identities(key, det)
        group = {m: gi for gi, grp in enumerate(rev.groups) for m in grp}
        conflict = set(rev.conflicts)
        lab_boxes = [np.asarray(b, float).reshape(-1, 4) for b in det["boxes"]]
        th = thresholds[fold_of[key]]
        for src in SOURCES:
            stills, owner = [], []
            boxes_of[src][key] = {"images": names, "boxes": []}
            for i, n in enumerate(names):
                lb = lab_boxes[i]
                if src == "labels":
                    b, sc, own = lb, np.ones(len(lb)), np.arange(len(lb))
                else:
                    pb, ps = preds[f"{key}/{n}"]
                    keep = ps >= th
                    b, sc = pb[keep], ps[keep]
                    own = match_to_labels(centres(b), centres(lb), box_len(lb)) if len(b) else []
                    own = np.asarray(own, int)
                boxes_of[src][key]["boxes"].append(np.round(b, 1).tolist())
                gsd = g["alt"][i] / g["fpx"][i]
                c = centres(b) if len(b) else np.zeros((0, 2))
                stills.append(
                    Still(times[i], gsd, c, ground(g, i, c) if len(c) else c, box_len(b) * gsd, sc)
                )
                owner.append(own)
            cand = candidates(stills, overlaps)
            y = np.full(len(cand), -1)
            for n in range(len(cand)):
                la = owner[cand.i[n]][cand.a[n]]
                lb_ = owner[cand.j[n]][cand.b[n]]
                if la < 0 and lb_ < 0:
                    continue
                if la < 0 or lb_ < 0:
                    y[n] = 0
                    continue
                ga, gb = group[(cand.i[n], la)], group[(cand.j[n], lb_)]
                if ga != gb:
                    y[n] = 0
                elif ga not in conflict:
                    y[n] = 1
            raw[src].append((key, cand, y, [len(s.score) for s in stills]))

    summary: dict = {
        "flights": len(keys),
        "features": list(FEATURES),
        "bootstrap": boot,
        "environment": env_report(),
    }
    out_dir = storage_path("processed", "phase3", create=True)
    rng = np.random.default_rng(boot["seed"])
    for src in SOURCES:
        rows = raw[src]
        flight_of = np.concatenate([np.full(len(c), k) for k, c, _, _ in rows])
        fold = np.array([fold_of[k] for k in flight_of])
        cand_all = {f: np.concatenate([getattr(c, f) for _, c, _, _ in rows]) for f in CAND_FIELDS}
        y = np.concatenate([yy for _, _, yy, _ in rows])
        motion_of = {}
        lab_rows = raw["labels"]
        for f in folds:  # grazing growth from linked label pairs of the other folds
            tr = [(c, yy) for k, c, yy, _ in lab_rows if fold_of[k] != f]
            d = np.concatenate([c.d_m[(yy == 1) & ~c.geotag] for c, yy in tr])
            dt = np.concatenate([c.dt_s[(yy == 1) & ~c.geotag] for c, yy in tr])
            motion_of[f] = MotionModel.fit(d, dt)
        x = {kind: np.zeros((len(y), len(FEATURES))) for kind in KINDS}
        start = 0
        for k, c, _, _ in rows:
            f = fold_of[k]
            for kind in KINDS:
                x[kind][start : start + len(c)] = features(c, motion_of[f], kind)
            start += len(c)
        log_odds = {kind: np.zeros(len(y)) for kind in KINDS}
        models = {}
        for kind in KINDS:
            models[kind] = {}
            for f in folds:
                tr = (fold != f) & (y >= 0)
                m = fit_logistic(x[kind][tr], y[tr])
                te = fold == f
                log_odds[kind][te] = m.log_odds(x[kind][te])
                models[kind][f] = {
                    "coef": m.coef.tolist(),
                    "mean": m.mean.tolist(),
                    "std": m.std.tolist(),
                }
        known = y >= 0
        flights = np.array(keys)
        idx = rng.integers(0, len(flights), (boot["resamples"], len(flights)))
        rows_of = {k: np.nonzero((flight_of == k) & known)[0] for k in keys}

        def boot_auc(
            s: np.ndarray,
            yy: np.ndarray,
            idx: np.ndarray = idx,
            rows_of: dict = rows_of,
            flights: np.ndarray = flights,
        ) -> list[float]:
            vals = []
            for draw in idx:
                r = np.concatenate([rows_of[k] for k in flights[draw]])
                vals.append(auc(s[r], yy[r]))
            v = np.array(vals)
            v = v[np.isfinite(v)]
            return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

        res = {
            "pairs": int(len(y)),
            "same_animal": int((y == 1).sum()),
            "different": int((y == 0).sum()),
            "unknown": int((y < 0).sum()),
            "motion_by_fold": {
                f: {"floor_m": m.floor_m, "diffusion_m2_s": m.diffusion_m2_s}
                for f, m in motion_of.items()
            },
            "models": models,
        }
        for kind in KINDS:
            res[f"auc_{kind}"] = {
                "value": auc(log_odds[kind][known], y[known] == 1),
                "ci95": boot_auc(log_odds[kind], y == 1),
            }
        summary[src] = res
        np.savez_compressed(
            out_dir / f"pairs_{src}.npz",
            flight=flight_of,
            y=y,
            log_odds_movement=log_odds["movement"],
            log_odds_fixed=log_odds["fixed"],
            x_movement=x["movement"],
            x_fixed=x["fixed"],
            counts=np.array(json.dumps({k: n for k, _, _, n in rows})),
            detections=np.array(json.dumps(boxes_of[src])),
            **cand_all,
        )
        print(
            f"{src}: {len(y)} pairs ({res['same_animal']} same, {res['different']} different, "
            f"{res['unknown']} unknown); out-of-fold AUC movement "
            f"{res['auc_movement']['value']:.3f} {np.round(res['auc_movement']['ci95'], 3)}, "
            f"fixed {res['auc_fixed']['value']:.3f} {np.round(res['auc_fixed']['ci95'], 3)}"
        )
    (REPO_ROOT / "results" / "phase3" / "pair_model.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
