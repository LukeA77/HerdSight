"""Gather the Phase 1 baseline evidence for each target (S1, S2, S3, S7) into one file (D-31).

Usage: python scripts/baseline_summary.py

Writes results/baselines/spec_baselines.json: for each measure, the best Phase 1 baseline, its 95%
interval where there is one, and the file it came from. The targets themselves are set by the
owner after reviewing these results and the baselines' failure cases, each with a stated reason,
in configs/specs.yaml, which is frozen (scripts/freeze.py phase1_specs) before any Phase 2 model is
trained or scored.

"Best baseline": D-FINE's out-of-fold value on every non-test image, unless zero-shot beats it on
the images both were run on. Each entry also carries "halve_the_gap", a reference point only (for
a score where 1 is perfect, b + (1 - b) / 2; for an error, b / 2); it is not a target.
"""

from __future__ import annotations

import json
import sys

from herdsight.utils.config import REPO_ROOT

RESULTS = REPO_ROOT / "results" / "baselines"


def load(name: str) -> dict | None:
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def entry(value: float, ci: list[float] | None, source: str, higher_is_better: bool) -> dict:
    ref = value + (1.0 - value) / 2.0 if higher_is_better else value / 2.0
    return {
        "baseline": round(value, 4),
        "baseline_ci95": [round(x, 4) for x in ci] if ci else None,
        "baseline_source": source,
        "better": "higher" if higher_is_better else "lower",
        "halve_the_gap": round(ref, 4),
    }


def main() -> int:
    dfine, zs = load("dfine_scores.json"), load("zeroshot_scores.json")
    if dfine is None:
        raise SystemExit("results/baselines/dfine_scores.json missing: run run_dfine_baselines.py")
    oof, shared = dfine["out_of_fold"], dfine.get("out_of_fold_on_zeroshot_sample")

    def best(path: tuple[str, ...], ci_key: str, higher: bool) -> dict:
        def get(o: dict | None, key: str) -> object:
            for p in path[:-1]:
                o = (o or {}).get(p)
            return (o or {}).get(key)

        if zs is not None and shared is not None:
            z, d = get(zs["out_of_fold"], path[-1]), get(shared, path[-1])
            if z is not None and d is not None and (z > d if higher else z < d):
                src = "zeroshot_scores.json: out_of_fold (beats D-FINE on the shared sample)"
                return entry(z, get(zs["out_of_fold"], ci_key), src, higher)
        v = get(oof, path[-1])
        if v is None:
            raise SystemExit(f"{'.'.join(path)} missing in dfine_scores.json")
        return entry(v, get(oof, ci_key), "dfine_scores.json: out_of_fold", higher)

    out: dict = {
        "note": "Baseline evidence for each target. Targets are set by the owner in "
        "configs/specs.yaml after review, then frozen before Phase 2 (D-31). halve_the_gap is "
        "a reference point only."
    }
    out["S1"] = {
        "centre_matched_recall": best(("centre_matched", "recall"), "recall_ci95", True),
        "centre_matched_precision": best(("centre_matched", "precision"), "precision_ci95", True),
        "centre_ap": best(("centre_ap", "ap_mean"), "ap_mean_ci95", True),
        "upright_ap50": best(("upright_ap", "ap50"), "ap50_ci95", True),
    }
    out["S2"] = {
        "count_mae": best(("counts", "mae"), "mae_ci95", False),
        "interval_90_coverage": {
            "baseline": round(oof["intervals"]["0.90"]["coverage"], 4),
            "baseline_ci95": oof["intervals"]["0.90"]["coverage_ci95"],
            "baseline_source": "dfine_scores.json: out_of_fold",
            "brief_target": [0.85, 0.95],
        },
    }
    s3 = load("dfine_s3.json")
    if s3 and s3.get("verified"):
        v = s3["verified"]
        m = min(
            ("naive", "geotag", "registration"), key=lambda k: v[f"{k}_mean_abs_relative_error"]
        )
        out["S3"] = {
            "flight_count_mean_abs_relative_error": {
                **entry(
                    v[f"{m}_mean_abs_relative_error"],
                    None,
                    f"dfine_s3.json: {m} on {v['flights']} verified flights",
                    False,
                ),
                "all_methods": {
                    k: v[f"{k}_mean_abs_relative_error"]
                    for k in ("naive", "geotag", "registration")
                },
            }
        }
    herds = load("dfine_onboard_herds.json")
    if herds:
        out["S7"] = {
            "herd_recall_found": entry(
                herds["herd_recall_found"],
                herds.get("herd_recall_found_ci95"),
                "dfine_onboard_herds.json",
                True,
            ),
            "herd_recall_grouped": entry(
                herds["herd_recall_grouped"],
                herds.get("herd_recall_grouped_ci95"),
                "dfine_onboard_herds.json",
                True,
            ),
            "latency_per_still_s": {
                "estimate": 2.0,
                "measured": False,
                "source": "D-29 estimate (no Raspberry Pi available, docs/questions.md Q8)",
            },
        }
    missing = [s for s in ("S1", "S2", "S3", "S7") if s not in out]
    out["missing"] = missing
    dest = RESULTS / "spec_baselines.json"
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
