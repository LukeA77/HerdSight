"""Fit the detection error model used on simulated flights (Phase 5 plan, step 4; D-11).

Usage: python scripts/fit_detection_model.py [--run dfine]

Rendered cattle are never scored, so home base sees simulated detections drawn from a model
fitted here on real images: the Phase 1 out-of-fold predictions ($HERDSIGHT_STORAGE/runs/
baselines/<run>/fold<k>.jsonl) on the cross-validation farms, at each fold's operating threshold
(results/baselines/dfine_scores.json, per_fold_fit). Predictions are matched to the labels with
the project's centre matching (configs/evaluation.yaml).

Writes results/data/detection_error_model.json:
  * detection: P(found) = logistic(b0 + b1 * log(length in pixels / 60) + b2 * crowded
    + u_group + v_image + w_animal), crowded = another labelled animal within one body length;
    u, v and w are random intercepts per farm-date group, per image and per animal (the same
    animal in overlapping stills of the S3 flights, from their candidate link components), so
    misses cluster as on real farms and an animal that is hard to see stays hard in every view.
    Fitted by penalised likelihood with the variances re-estimated in turn (a Laplace-type
    approximation), with scipy only;
  * false detections per hectare of ground in view, overall and per group (their spread sets how
    many fixed false-detection objects a simulated world gets);
  * position error of found animals, as a share of body length.
Also reports the fit for the worst group alone (a farm like Derval, D-33), for the "new farm"
case. Calves and partly hidden animals are not in the labels: they are assumptions (plan, step 4).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from herdsight.data.icaerus import default_root, iter_images
from herdsight.data.splits import FlightStats
from herdsight.evaluate.centre_match import box_centres, match_hungarian, short_side
from herdsight.geo.nadir import focal_px_from_35mm
from herdsight.utils.config import REPO_ROOT, load_config
from herdsight.utils.paths import storage_path

LENGTH_REF_PX = 60.0


def load_rows(run: str) -> tuple[list[dict], dict]:
    sp = json.loads((REPO_ROOT / "results" / "splits.json").read_text())["icaerus_v2"]["flights"]
    scores = json.loads((REPO_ROOT / "results" / "baselines" / f"{run}_scores.json").read_text())
    thr = {k: v["threshold"] for k, v in scores["per_fold_fit"].items()}
    meta = {
        f"{r['farm']}/{r['flight']}/{r['image']}": r
        for r in csv.DictReader((REPO_ROOT / "results" / "data" / "icaerus_images.csv").open())
    }
    preds = {}
    for k in thr:
        path = storage_path("runs", "baselines", run, f"{k}.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            preds[d["image"]] = (k, d)
    ev = load_config("evaluation")
    kappa, sim = float(ev.require("centre_kappa")), float(ev.require("centre_match_similarity"))
    # the same animal in overlapping stills (S3 flights' link components): one animal effect
    same_animal: dict[tuple[str, int], str] = {}
    for path in sorted(storage_path("processed", "s3_links").glob("*.json")):
        ln = json.loads(path.read_text(encoding="utf-8"))
        for c, members in enumerate(ln["components"]):
            for im, bx in members:
                key = f"{ln['farm']}/{ln['flight']}/{ln['images'][im]}"
                same_animal[(key, int(bx))] = f"{ln['farm']}/{ln['flight']}#{c}"
    animals, images = [], []
    for s, problem in iter_images(default_root(), with_metadata=False):
        key = f"{s.farm}/{s.flight}/{s.path.name}"
        fold = sp.get(f"{s.farm}/{s.flight}")
        if not fold or not fold.startswith("fold") or problem or not s.labelled or key not in preds:
            continue
        k, d = preds[key]
        b = np.asarray(d["boxes"], float).reshape(-1, 4)
        sc = np.asarray(d["scores"], float)
        b = b[sc >= thr[k]]
        t = np.asarray(s.boxes_xyxy, float).reshape(-1, 4)
        group = FlightStats(s.farm, s.flight, 0, 0).group
        m = meta.get(key, {})
        f_px = focal_px_from_35mm(float(m.get("focal_35mm") or 24), s.width, s.height)
        gsd = float(m.get("rel_alt_m") or 0) / f_px if m.get("rel_alt_m") else None
        found = np.zeros(len(t), bool)
        err = []
        if len(t):
            tc, pc = box_centres(t), box_centres(b)
            mt = match_hungarian(pc, tc, np.maximum(short_side(t), 1.0), kappa, sim)
            found[mt.pairs[:, 1]] = True
            length = np.maximum(t[:, 2] - t[:, 0], t[:, 3] - t[:, 1])
            for pi, ti in mt.pairs:
                err.append(float(np.linalg.norm(pc[pi] - tc[ti]) / length[ti]))
            dist = np.linalg.norm(tc[:, None] - tc[None], axis=2) + np.eye(len(t)) * 1e9
            crowded = (dist < length[:, None]).any(1)
            for i in range(len(t)):
                animals.append(
                    {
                        "animal": same_animal.get((key, i), f"{key}#{i}"),
                        "image": key,
                        "group": group,
                        "length_px": float(length[i]),
                        "crowded": bool(crowded[i]),
                        "found": bool(found[i]),
                    }
                )
            n_fp = len(b) - len(mt.pairs)
        else:
            n_fp = len(b)
        area_ha = (s.width * gsd) * (s.height * gsd) / 1e4 if gsd else None
        images.append(
            {"image": key, "group": group, "fp": int(n_fp), "area_ha": area_ha, "err": err}
        )
    return animals, {"images": images}


def fit_glmm(
    y: NDArray, x: NDArray, effects: list[tuple[str, NDArray, int]], iters: int = 10
) -> dict:
    """Logistic regression with random intercepts (one per entry of `effects`: name, index per
    row, number of levels), by penalised likelihood with the variances re-estimated in turn
    (posterior variances from the Laplace approximation added to the moment estimate)."""
    p = x.shape[1]
    sizes = [n for _, _, n in effects]
    offs = np.cumsum([p, *sizes])
    var = [0.5] * len(effects)
    theta = np.zeros(offs[-1])

    def split(th: NDArray) -> tuple[NDArray, list[NDArray]]:
        return th[:p], [th[offs[k] : offs[k + 1]] for k in range(len(effects))]

    def eta_of(th: NDArray) -> NDArray:
        beta, us = split(th)
        return x @ beta + sum(u[idx] for u, (_, idx, _) in zip(us, effects, strict=True))

    def nll(th: NDArray, var: list[float]) -> float:
        eta = eta_of(th)
        _, us = split(th)
        pen = sum(0.5 * np.sum(u**2) / v for u, v in zip(us, var, strict=True))
        return -(np.sum(y * eta - np.logaddexp(0, eta)) - pen)

    def grad(th: NDArray, var: list[float]) -> NDArray:
        r = y - 1 / (1 + np.exp(-eta_of(th)))
        _, us = split(th)
        parts = [x.T @ r]
        for u, v, (_, idx, n) in zip(us, var, effects, strict=True):
            parts.append(np.bincount(idx, r, n) - u / v)
        return -np.concatenate(parts)

    for _ in range(iters):
        theta = minimize(nll, theta, jac=grad, args=(var,), method="L-BFGS-B").x
        mu = 1 / (1 + np.exp(-eta_of(theta)))
        w = mu * (1 - mu)
        _, us = split(theta)
        var = [
            max(float(np.mean(u**2 + 1 / (np.bincount(idx, w, n) + 1 / v))), 1e-4)
            for u, v, (_, idx, n) in zip(us, var, effects, strict=True)
        ]
    beta, us = split(theta)
    out = {
        "coefficients": dict(
            zip(("intercept", "log_length_over_60px", "crowded"), map(float, beta), strict=True)
        ),
    }
    for (name, _, _), v, u in zip(effects, var, us, strict=True):
        out[f"{name}_sd"] = float(np.sqrt(v))
        out[f"{name}_effects"] = u.tolist()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="dfine")
    args = ap.parse_args(argv)
    animals, extra = load_rows(args.run)
    images = extra["images"]
    groups = sorted({a["group"] for a in animals})
    gi = {g: i for i, g in enumerate(groups)}
    ims = sorted({a["image"] for a in animals})
    ii = {m: i for i, m in enumerate(ims)}
    y = np.array([a["found"] for a in animals], float)
    x = np.column_stack(
        [
            np.ones(len(animals)),
            np.log(np.array([a["length_px"] for a in animals]) / LENGTH_REF_PX),
            np.array([a["crowded"] for a in animals], float),
        ]
    )
    g_idx = np.array([gi[a["group"]] for a in animals])
    i_idx = np.array([ii[a["image"]] for a in animals])
    names = sorted({a["animal"] for a in animals})
    ai = {n: i for i, n in enumerate(names)}
    a_idx = np.array([ai[a["animal"]] for a in animals])
    multi = int(np.sum(np.bincount(a_idx) > 1))
    fit = fit_glmm(
        y,
        x,
        [("group", g_idx, len(groups)), ("image", i_idx, len(ims)), ("animal", a_idx, len(names))],
    )
    fit["group_effects"] = dict(zip(groups, fit["group_effects"], strict=True))
    del fit["image_effects"], fit["animal_effects"]
    fit["animals_seen_in_several_stills"] = multi
    recall_by_group = {g: float(y[g_idx == gi[g]].mean()) for g in groups}
    worst = min(recall_by_group, key=recall_by_group.get)
    fp_by_group: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
    for im in images:
        if im["area_ha"]:
            fp_by_group[im["group"]][0] += im["fp"]
            fp_by_group[im["group"]][1] += im["area_ha"]
    fp_rate = {g: v[0] / v[1] for g, v in fp_by_group.items() if v[1] > 0}
    tot_fp = sum(v[0] for v in fp_by_group.values())
    tot_ha = sum(v[1] for v in fp_by_group.values())
    errs = np.array([e for im in images for e in im["err"]])
    out = {
        "run": args.run,
        "animals": len(animals),
        "images_with_animals": len(ims),
        "images_scored": len(images),
        "groups": len(groups),
        "recall_raw": float(y.mean()),
        "recall_by_group": recall_by_group,
        "detection": fit,
        "length_ref_px": LENGTH_REF_PX,
        "new_farm_case": {
            "group": worst,
            "recall": recall_by_group[worst],
            "group_effect": fit["group_effects"][worst],
            "note": "for the 'new farm' sweep, use this group's effect for every animal",
        },
        "false_detections_per_ha": {
            "overall": tot_fp / tot_ha if tot_ha else None,
            "by_group": fp_rate,
            "images_with_ground_size": sum(1 for im in images if im["area_ha"]),
        },
        "position_error_share_of_length": {
            "median": float(np.median(errs)) if len(errs) else None,
            "p90": float(np.quantile(errs, 0.9)) if len(errs) else None,
            "rms": float(np.sqrt(np.mean(errs**2))) if len(errs) else None,
        },
        "assumptions_not_fitted": [
            "calves: detection probability extrapolated by size from adults (no calf labels)",
            "partly hidden animals: detection probability scaled by visible share (no labels)",
        ],
        "note": "Phase 1 out-of-fold predictions on real images; refit after Phase 2",
    }
    dest = REPO_ROOT / "results" / "data" / "detection_error_model.json"
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "detection"}, indent=1))
    print(json.dumps({k: v for k, v in fit.items() if k != "group_effects"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
