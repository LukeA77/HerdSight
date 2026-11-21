"""Where and why the D-FINE home-base baseline fails (input to the D-31 target review).

Usage: python scripts/failure_review.py [--detector dfine]

Uses the same out-of-fold predictions, per-fold score thresholds and IoU matching as
scripts/score_baseline.py, then breaks the errors down:
  * missed animals: by size, crowding, image edge, altitude and farm-date group, and whether the
    animal was found below the score cut-off, found but badly placed, or not found at all;
  * false detections: duplicates of a found animal, badly placed boxes next to a missed animal,
    detections on images with no cattle, and other background; how concentrated they are;
  * centre placement of correctly found animals, compared with the hand-labelling spread (kappa);
  * per farm-date group results and the images with the largest count errors.
Writes results/baselines/<detector>_failures.json. Diagnostic only: nothing here is used to
choose a model, threshold or target automatically.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from herdsight.data.icaerus import default_root, iter_images
from herdsight.data.splits import FlightStats
from herdsight.evaluate.centre_match import box_centres, short_side
from herdsight.evaluate.upright_ap import iou_matrix
from herdsight.utils.config import REPO_ROOT, load_config
from herdsight.utils.paths import storage_path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_baseline import cross_fit, load_preds  # noqa: E402

NEAR_IOU = 0.1  # a box overlapping a missed animal this much counts as "badly placed"
DUP_IOU = 0.3  # a false box overlapping a found animal this much counts as a duplicate


def match(pred: np.ndarray, true: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one IoU matching (as score_baseline.match_iou). Returns matched flags."""
    pm, tm = np.zeros(len(pred), bool), np.zeros(len(true), bool)
    m = iou_matrix(pred, true)
    if m.size:
        w = np.where(m >= t, m, 0.0)
        r, c = linear_sum_assignment(w, maximize=True)
        ok = w[r, c] >= t
        pm[r[ok]], tm[c[ok]] = True, True
    return pm, tm


def rate(flags: list[bool]) -> dict:
    return {"n": len(flags), "miss_rate": round(float(np.mean(flags)), 4) if flags else None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="dfine")
    args = ap.parse_args(argv)
    ev = load_config("evaluation")
    iou_t = float(ev.require("iou_match_threshold"))
    kappa = float(ev.require("centre_kappa"))
    sim_t = float(ev.require("centre_match_similarity"))
    sp = json.loads((REPO_ROOT / "results" / "splits.json").read_text())["icaerus_v2"]
    folds = [f"fold{k}" for k in range(sp["n_folds"])]
    runs = storage_path("runs", "baselines", args.detector)
    preds: dict = {}
    for f in folds:
        preds |= load_preds(runs / f"{f}.jsonl")
    truth, fold_of, group_of = {}, {}, {}
    for s, problem in iter_images(default_root(), with_metadata=False):
        fold = sp["flights"].get(f"{s.farm}/{s.flight}")
        if problem is None and s.labelled and fold in folds:
            k = f"{s.farm}/{s.flight}/{s.path.name}"
            truth[k], fold_of[k] = s.boxes_xyxy, fold
            group_of[k] = FlightStats(s.farm, s.flight, 0, 0).group
    meta = {}
    with (REPO_ROOT / "results" / "data" / "icaerus_images.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            meta[f"{r['farm']}/{r['flight']}/{r['image']}"] = r
    keys = sorted(k for k in preds if k in truth)
    start, stop, step = ev.require("operating_threshold_grid")
    grid = np.round(np.arange(start, stop + 1e-9, step), 4)
    fits = cross_fit(keys, fold_of, preds, truth, grid)

    animals = []  # one row per true animal
    fps = []  # one row per false detection
    tp_scores, offsets = [], []
    per_image = []
    for k in keys:
        boxes, scores = preds[k]
        true = truth[k]
        thr = fits[fold_of[k]]["threshold"]
        keep = scores >= thr
        pb, ps = boxes[keep], scores[keep]
        pm, tm = match(pb, true, iou_t)
        m_all = iou_matrix(boxes, true)
        m_kept = iou_matrix(pb, true)
        tt = iou_matrix(true, true)
        np.fill_diagonal(tt, 0.0) if tt.size else None
        w, h = float(meta[k]["width"]), float(meta[k]["height"])
        alt = float(meta[k]["rel_alt_m"]) if meta[k]["rel_alt_m"] else math.nan
        ss = short_side(true) if len(true) else np.zeros(0)
        for j, b in enumerate(true):
            if tm[j]:
                kind = "found"
            elif m_all.size and (m_all[:, j] >= iou_t).any():
                kind = "below_cutoff"
            elif m_kept.size and (m_kept[:, j] >= NEAR_IOU).any():
                kind = "badly_placed"
            else:
                kind = "not_found"
            animals.append(
                {
                    "kind": kind,
                    "short_px": float(ss[j]),
                    "touching": bool(tt.size and (tt[j] > 0).any()),
                    "edge": bool(b[0] <= 2 or b[1] <= 2 or b[2] >= w - 2 or b[3] >= h - 2),
                    "alt": alt,
                    "group": group_of[k],
                }
            )
        # centre placement of found animals (best IoU partner among matched boxes)
        if pm.any():
            sub = m_kept[np.ix_(pm, tm)]
            r, c = linear_sum_assignment(sub, maximize=True)
            pc = box_centres(pb[pm][r])
            tc = box_centres(true[tm][c])
            d = np.linalg.norm(pc - tc, axis=1) / np.maximum(short_side(true[tm][c]), 1.0)
            offsets.extend(d.tolist())
            tp_scores.extend(ps[pm].tolist())
        found_box = pb[pm]
        for i in np.nonzero(~pm)[0]:
            if len(true) == 0:
                kind = "empty_image"
            elif len(found_box) and iou_matrix(pb[i : i + 1], found_box).max() >= DUP_IOU:
                kind = "duplicate"
            elif (~tm).any() and m_kept[i, ~tm].max() >= NEAR_IOU:
                kind = "badly_placed"
            else:
                kind = "background"
            fps.append({"kind": kind, "score": float(ps[i]), "image": k, "group": group_of[k]})
        per_image.append(
            {
                "image": k,
                "group": group_of[k],
                "alt": alt,
                "true": len(true),
                "pred": int(keep.sum()),
                "fp": int((~pm).sum()),
                "missed": int((~tm).sum()),
            }
        )

    out: dict = {
        "detector": args.detector,
        "note": "Diagnostic breakdown of out-of-fold errors at each fold's operating threshold. "
        "Public-data results, not field results.",
        "n_images": len(keys),
        "n_animals": len(animals),
    }
    kinds = Counter(a["kind"] for a in animals)
    out["animals"] = {"by_outcome": dict(kinds)}
    missed = [a for a in animals if a["kind"] != "found"]
    out["animals"]["missed"] = len(missed)
    sizes = np.array([a["short_px"] for a in animals])
    edges = np.quantile(sizes, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    by_size = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = [a["kind"] != "found" for a in animals if lo <= a["short_px"] <= hi]
        by_size.append({"short_side_px": [round(lo, 1), round(hi, 1)], **rate(sel)})
    out["animals"]["miss_rate_by_size_quintile"] = by_size
    for key in ("touching", "edge"):
        out["animals"][f"miss_rate_by_{key}"] = {
            str(v): rate([a["kind"] != "found" for a in animals if a[key] == v])
            for v in (True, False)
        }
    alts = defaultdict(list)
    for a in animals:
        if math.isfinite(a["alt"]):
            alts[f"{round(a['alt'] / 10) * 10:.0f}m"].append(a["kind"] != "found")
    out["animals"]["miss_rate_by_altitude"] = {k: rate(v) for k, v in sorted(alts.items())}
    grp = defaultdict(list)
    for a in animals:
        grp[a["group"]].append(a["kind"] != "found")
    out["animals"]["miss_rate_by_group"] = {k: rate(v) for k, v in sorted(grp.items())}

    fk = Counter(f["kind"] for f in fps)
    out["false_detections"] = {"total": len(fps), "by_kind": dict(fk)}
    empty_imgs = [p for p in per_image if p["true"] == 0]
    with_fp = sorted((p for p in empty_imgs if p["fp"]), key=lambda p: -p["fp"])
    tot = sum(p["fp"] for p in empty_imgs)
    out["false_detections"]["empty_images"] = {
        "images": len(empty_imgs),
        "images_with_any": len(with_fp),
        "detections": tot,
        "share_from_worst_10_images": round(sum(p["fp"] for p in with_fp[:10]) / max(tot, 1), 4),
        "per_group": {
            g: {
                "images": sum(1 for p in empty_imgs if p["group"] == g),
                "fp_per_image": round(
                    float(np.mean([p["fp"] for p in empty_imgs if p["group"] == g])), 4
                ),
            }
            for g in sorted({p["group"] for p in empty_imgs})
        },
        "worst_images": [{k: p[k] for k in ("image", "group", "fp")} for p in with_fp[:10]],
    }
    fp_s = np.array([f["score"] for f in fps])
    tp_s = np.array(tp_scores)
    out["false_detections"]["score_share_at_least_0.9"] = {
        "false": round(float((fp_s >= 0.9).mean()), 4) if fp_s.size else None,
        "true": round(float((tp_s >= 0.9).mean()), 4) if tp_s.size else None,
    }

    d = np.array(offsets)
    sim = np.exp(-(d**2) / (2 * kappa**2))
    out["centre_placement_of_found_animals"] = {
        "n": int(d.size),
        "offset_over_short_side": {
            "median": round(float(np.median(d)), 4),
            "p90": round(float(np.quantile(d, 0.9)), 4),
            "rms": round(float(np.sqrt(np.mean(d**2))), 4),
        },
        "hand_label_rms_kappa": kappa,
        "share_failing_centre_match": round(float((sim < sim_t).mean()), 4),
        "explains": "Animals found by box overlap but whose box centre is further from the "
        "labelled centre than the centre-match rule allows; this is the gap between box-overlap "
        "recall and centre-matched recall.",
    }

    groups = {}
    for g in sorted({p["group"] for p in per_image}):
        ps_ = [p for p in per_image if p["group"] == g]
        t = np.array([p["true"] for p in ps_], float)
        pr = np.array([p["pred"] for p in ps_], float)
        alts_g = [p["alt"] for p in ps_ if math.isfinite(p["alt"])]
        groups[g] = {
            "images": len(ps_),
            "animals": int(t.sum()),
            "median_alt_m": round(float(np.median(alts_g)), 1) if alts_g else None,
            "count_mae": round(float(np.mean(np.abs(pr - t))), 4),
            "mean_signed_error": round(float(np.mean(pr - t)), 4),
            "missed": int(sum(p["missed"] for p in ps_)),
            "false": int(sum(p["fp"] for p in ps_)),
        }
    out["by_group"] = groups
    worst = sorted(per_image, key=lambda p: -abs(p["pred"] - p["true"]))[:15]
    out["worst_count_images"] = [
        {k: p[k] for k in ("image", "group", "true", "pred", "missed", "fp")} for p in worst
    ]
    abs_err = np.array([abs(p["pred"] - p["true"]) for p in per_image], float)
    srt = np.sort(abs_err)[::-1]
    out["count_error_concentration"] = {
        "total_abs_error": int(abs_err.sum()),
        "share_from_worst_5pct_images": round(
            float(srt[: max(1, len(srt) // 20)].sum() / max(abs_err.sum(), 1)), 4
        ),
        "images_exact": int((abs_err == 0).sum()),
    }
    dest = REPO_ROOT / "results" / "baselines" / f"{args.detector}_failures.json"
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
