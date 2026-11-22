"""Phase 3 H25: does appearance improve the same-animal model?

Usage: python scripts/phase3_h25.py   (after scripts/phase3_pairs.py and phase3_appearance.py)

Checks the Phase 3 freeze first. For every candidate pair, the distance between the two
detections' descriptors (colour_texture: chi-squared; pretrained: cosine) is added to the
"fixed" feature set of scripts/phase3_pairs.py, and the logistic model is refitted out of fold
(two folds fit, the third is scored). The out-of-fold area under the ROC curve with and without
appearance is compared on the same pairs, with a paired bootstrap over flights.

Fixed before the first run (2026-11-22): H25 is judged on the Phase 1 detector's pairs (as H23);
labels are reported beside them. It holds if either descriptor raises the AUC by at least
auc_gain_min with the paired interval excluding zero (configs/phase3.yaml). Writes
results/phase3/h25.json.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from herdsight.count.evidence import auc, fit_logistic
from herdsight.utils.config import REPO_ROOT, load_config
from herdsight.utils.env_report import env_report
from herdsight.utils.paths import storage_path

SOURCES = ("dfine", "labels")
DESCRIPTORS = ("colour_texture", "pretrained")


def chi2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 0.5 * np.sum((a - b) ** 2 / np.maximum(a + b, 1e-12), axis=1)


def main() -> int:
    chk = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "freeze.py"),
            "--phase",
            "phase3_prereg",
            "--check",
        ],
        capture_output=True,
        text=True,
    )
    print(chk.stdout.strip())
    if chk.returncode != 0:
        return 1
    p3 = load_config("phase3")
    boot = p3.require("bootstrap")
    rule = p3.require("hypotheses")["H25"]
    sp = json.loads((REPO_ROOT / p3.require("folds")).read_text())["icaerus_v2"]["flights"]
    d3 = storage_path("processed", "phase3")
    out: dict = {"environment": env_report(), "bootstrap": boot, "rule": rule}
    for src in SOURCES:
        d = np.load(d3 / f"pairs_{src}.npz")
        desc = json.loads(str(np.load(d3 / f"appearance_{src}.npz")["descriptors"]))
        flight = d["flight"]
        y = d["y"]
        fold = np.array([sp[k] for k in flight])
        x0 = d["x_fixed"]
        dist = {}
        for name in DESCRIPTORS:
            va = np.array(
                [desc[k][i][name][a] for k, i, a in zip(flight, d["i"], d["a"], strict=True)]
            )
            vb = np.array(
                [desc[k][j][name][b] for k, j, b in zip(flight, d["j"], d["b"], strict=True)]
            )
            dist[name] = chi2(va, vb) if name == "colour_texture" else 1.0 - np.sum(va * vb, 1)
        sets = {"none": x0, **{n: np.column_stack([x0, dist[n]]) for n in DESCRIPTORS}}
        lo = {}
        for name, x in sets.items():
            lo[name] = np.zeros(len(y))
            for f in sorted(set(fold)):
                tr = (fold != f) & (y >= 0)
                m = fit_logistic(x[tr], y[tr])
                lo[name][fold == f] = m.log_odds(x[fold == f])
        known = y >= 0
        keys = np.array(sorted(set(flight)))
        rows_of = {k: np.nonzero((flight == k) & known)[0] for k in keys}
        rng = np.random.default_rng(boot["seed"])
        draws = rng.integers(0, len(keys), (boot["resamples"], len(keys)))
        base = auc(lo["none"][known], y[known] == 1)
        res = {"pairs_known": int(known.sum()), "auc_without_appearance": base}
        for name in DESCRIPTORS:
            val = auc(lo[name][known], y[known] == 1)
            diffs = []
            for dr in draws:
                r = np.concatenate([rows_of[k] for k in keys[dr]])
                diffs.append(auc(lo[name][r], y[r] == 1) - auc(lo["none"][r], y[r] == 1))
            v = np.array(diffs)
            v = v[np.isfinite(v)]
            ci = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            same, diff_ = dist[name][y == 1], dist[name][y == 0]
            res[name] = {
                "auc": val,
                "gain": val - base,
                "gain_ci95": ci,
                "holds": bool(val - base >= rule["auc_gain_min"] and ci[0] > 0),
                "distance_median_same_animal": float(np.median(same)),
                "distance_median_different": float(np.median(diff_)),
                "auc_of_distance_alone": auc(-dist[name][known], y[known] == 1),
            }
            print(
                f"{src} {name}: AUC {val:.4f} vs {base:.4f} without, gain {val - base:+.4f} "
                f"{np.round(ci, 4)}; distance alone AUC {res[name]['auc_of_distance_alone']:.3f}"
            )
        out[src] = res
    out["H25"] = {
        "judged_on": "dfine",
        "holds": any(out["dfine"][n]["holds"] for n in DESCRIPTORS),
        "by_descriptor": {n: out["dfine"][n]["holds"] for n in DESCRIPTORS},
    }
    print("H25:", out["H25"])
    (REPO_ROOT / "results" / "phase3" / "h25.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
