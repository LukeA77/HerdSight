"""Phase 3 step 4 on real flights: global grouping against greedy joining (H23, H24).

Usage: python scripts/phase3_grouping.py     (after scripts/phase3_pairs.py)

Checks the Phase 3 freeze first (scripts/freeze.py --phase phase3_prereg --check) and stops if
anything changed. For each verified flight and each source of detections (labels; the Phase 1
detector, out of fold), the out-of-fold log odds of every candidate pair
($HERDSIGHT_STORAGE/processed/phase3/pairs_<source>.npz) are grouped four ways:

  greedy_fixed / global_fixed        evidence with distance scaled by the Phase 1 gate
  greedy_movement / global_movement  the same gate widened by the grazing allowance

Choices fixed before this script was first run (2026-11-22), from out-of-fold pair results
only (results/phase3/pair_model.json), never from a flight count:
* H23 is judged with the Phase 1 detector's detections (what S3 measures) and the "fixed"
  evidence (higher out-of-fold pair AUC); labels are reported beside it.
* H24 compares global_movement with global_fixed on the flights with at least one pair of
  overlapping stills more than separate_pass_gap_s apart (configs/phase3.yaml); real flights give
  the direction only.
The Phase 1 method's errors (results/phase3/error_split.json inputs) are reported as a reference.
Only the real-flight part of each hypothesis is judged here; the simulated part follows.

Also reported, per method: the share of detections in exactly solved parts, the heuristic's gap
to the optimum there, and pair-level linking precision and recall on candidate pairs whose truth
is known. Writes results/phase3/grouping_real.json.

Re-run for another detector's out-of-fold predictions with --detector <run> (for example
dfine_ref, the final detector); its files get the run name as a suffix and the frozen Phase 3
results stay as they are.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict

import numpy as np

from herdsight.count.clustering import cluster, greedy_join
from herdsight.utils.config import REPO_ROOT, load_config
from herdsight.utils.env_report import env_report
from herdsight.utils.paths import storage_path

KINDS = ("fixed", "movement")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="dfine")
    det_run = ap.parse_args(argv).detector
    SOURCES = {"labels": "oracle_s3_v20.json", det_run: f"{det_run}_s3_v20.json"}  # noqa: N806
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
    exact = p3.require("exact_solver")
    gap_s = float(p3.require("separate_pass_gap_s"))
    rules = p3.require("hypotheses")
    truth = json.loads((REPO_ROOT / p3.require("truth")).read_text())["flights"]
    rng = np.random.default_rng(boot["seed"])
    out: dict = {"environment": env_report(), "bootstrap": boot, "exact_solver": exact}
    for src, s3_file in SOURCES.items():
        d = np.load(
            storage_path("processed", "phase3")
            / (
                f"pairs_labels_{det_run}.npz"
                if src == "labels" and det_run != "dfine"
                else f"pairs_{src}.npz"
            )
        )
        counts = json.loads(str(d["counts"]))
        phase1 = json.loads((REPO_ROOT / "results" / "phase3" / s3_file).read_text())["flights"]
        keys = sorted(counts)
        flights: dict = {}
        link_tally: dict = defaultdict(lambda: np.zeros(3))  # true pos, false pos, false neg
        for key in keys:
            m = d["flight"] == key
            i, a, j, b, y = d["i"][m], d["a"][m], d["j"][m], d["b"][m], d["y"][m]
            dt = d["dt_s"][m]
            true_n = truth[key]["verified_unique"]
            rec = {
                "truth": true_n,
                "separate_passes": bool((dt > gap_s).any()),
                "phase1": phase1[key]["estimates"]["registration"],
            }
            for kind in KINDS:
                lo = d[f"log_odds_{kind}"][m]
                pairs = [
                    ((int(i[n]), int(a[n])), (int(j[n]), int(b[n])), float(lo[n]))
                    for n in range(len(lo))
                ]
                g = greedy_join(counts[key], pairs)
                c = cluster(counts[key], pairs, exact["max_detections"], exact["time_limit_s"])
                rec[f"greedy_{kind}"] = len(g)
                rec[f"global_{kind}"] = len(c.groups)
                n_nodes = max(sum(counts[key]), 1)
                rec[f"global_{kind}_exact_share"] = c.nodes_in_exact_parts / n_nodes
                rec[f"global_{kind}_heuristic_gap"] = c.heuristic_gap
                for name, groups in ((f"greedy_{kind}", g), (f"global_{kind}", c.groups)):
                    gid = {node: k for k, grp in enumerate(groups) for node in grp}
                    together = np.array(
                        [
                            gid[(int(i[n]), int(a[n]))] == gid[(int(j[n]), int(b[n]))]
                            for n in range(len(y))
                        ],
                        bool,
                    )
                    known = y >= 0
                    link_tally[name] += [
                        (together & (y == 1)).sum(),
                        (together & (y == 0)).sum(),
                        (~together & (y == 1) & known).sum(),
                    ]
            flights[key] = rec
        methods = ["phase1"] + [f"{g}_{k}" for k in KINDS for g in ("greedy", "global")]
        err = {
            mth: np.array(
                [(flights[k][mth] - flights[k]["truth"]) / flights[k]["truth"] for k in keys]
            )
            for mth in methods
        }
        idx = rng.integers(0, len(keys), (boot["resamples"], len(keys)))

        def ci(v: np.ndarray, idx: np.ndarray = idx) -> list[float]:
            b = v[idx].mean(axis=1)
            return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

        summary = {
            mth: {
                "mare": float(np.abs(e).mean()),
                "mare_ci95": ci(np.abs(e)),
                "mean_signed": float(e.mean()),
            }
            for mth, e in err.items()
        }
        for name, (tp, fp, fn) in link_tally.items():
            summary[name]["link_precision"] = float(tp / max(tp + fp, 1))
            summary[name]["link_recall"] = float(tp / max(tp + fn, 1))
        comp = {}
        for kind in KINDS:
            diff = np.abs(err[f"global_{kind}"]) - np.abs(err[f"greedy_{kind}"])
            comp[f"global_minus_greedy_{kind}"] = {"mean": float(diff.mean()), "ci95": ci(diff)}
        sub = np.array([flights[k]["separate_passes"] for k in keys])
        h24 = None
        if sub.any():
            dsub = np.abs(err["global_movement"][sub]) - np.abs(err["global_fixed"][sub])
            h24 = {"flights": int(sub.sum()), "movement_minus_fixed_mean": float(dsub.mean())}
        out[src] = {"flights": flights, "summary": summary, "paired": comp, "h24_subset": h24}
        print(src)
        for mth in methods:
            s = summary[mth]
            extra = (
                f", links precision {s['link_precision']:.3f} recall {s['link_recall']:.3f}"
                if "link_precision" in s
                else ""
            )
            print(
                f"  {mth:16s} MARE {s['mare']:.3f} {np.round(s['mare_ci95'], 3)} "
                f"signed {s['mean_signed']:+.3f}{extra}"
            )
        for k, v in comp.items():
            print(f"  {k}: {v['mean']:+.3f} {np.round(v['ci95'], 3)}")
        print(f"  H24 subset: {h24}")
    r = rules["H23"]
    c = out[det_run]["paired"]["global_minus_greedy_fixed"]
    out["H23_real"] = {
        "detections": det_run,
        "evidence": "fixed",
        "gain": -c["mean"],
        "gain_ci95": [-c["ci95"][1], -c["ci95"][0]],
        "gain_min": r["mare_gain_min"],
        "real_part_holds": bool(-c["mean"] >= r["mare_gain_min"] and c["ci95"][1] < 0),
        "note": "real-flight part only; the simulated non-inferiority part is judged separately",
    }
    h = out[det_run]["h24_subset"]
    out["H24_real_direction"] = (
        None if h is None else {"movement_better": h["movement_minus_fixed_mean"] < 0, **h}
    )
    print("H23 (real part):", out["H23_real"])
    print("H24 (real direction):", out["H24_real_direction"])
    (
        REPO_ROOT
        / "results"
        / "phase3"
        / ("grouping_real.json" if det_run == "dfine" else f"grouping_real_{det_run}.json")
    ).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
