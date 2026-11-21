"""Judge the Phase 1 hypotheses (docs/hypotheses.md, H1-H6) from the scored baselines.

Usage: python scripts/judge_hypotheses.py [--dir results/baselines] [--out <json>]

Each rule below is the hypothesis's written prediction, turned into code before the fine-tuned
baseline was scored. The outcome is "held", "not held" or "not tested" (with the reason); the
values it rests on, and their 95% intervals, are written next to it. Nothing is judged on images
or flights the hypothesis does not name.

Which detector: H1 compares the fine-tuned D-FINE baseline with zero-shot on the images both were
run on. H2-H5 are about the fine-tuned baseline (D-FINE, out-of-fold on every non-test image).
H6 uses D-FINE's per-image counts on the flights with owner-verified identities.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from herdsight.utils.config import REPO_ROOT


def load(d: Path, name: str) -> dict | None:
    p = d / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def outcome(ok: bool | None, why_not_tested: str = "") -> dict:
    if ok is None:
        return {"outcome": "not tested", "reason": why_not_tested}
    return {"outcome": "held" if ok else "not held"}


def band(rows: list[dict], lo: int) -> dict | None:
    return next((r for r in rows if r["band"][0] == lo), None)


def h1(dfine: dict, zs: dict | None) -> dict:
    rule = (
        "On the zero-shot image sample: D-FINE centre AP >= zero-shot centre AP + 0.2, D-FINE "
        "count MAE <= half of zero-shot's, and for both measures the two 95% intervals do not "
        "overlap."
    )
    d = dfine.get("out_of_fold_on_zeroshot_sample")
    if zs is None or d is None:
        return {"rule": rule, **outcome(None, "zero-shot or shared-sample scores missing")}
    z = zs["out_of_fold"]
    ap_d, ap_z = d["centre_ap"]["ap_mean"], z["centre_ap"]["ap_mean"]
    ci_ap_d, ci_ap_z = d["centre_ap"].get("ap_mean_ci95"), z["centre_ap"].get("ap_mean_ci95")
    mae_d, mae_z = d["counts"]["mae"], z["counts"]["mae"]
    ci_mae_d, ci_mae_z = d["counts"]["mae_ci95"], z["counts"]["mae_ci95"]
    if None in (ci_ap_d, ci_ap_z, ci_mae_d, ci_mae_z):
        return {"rule": rule, **outcome(None, "an interval is missing: rescore both detectors")}
    checks = {
        "ap_gain_at_least_0.2": ap_d - ap_z >= 0.2,
        "mae_at_most_half": mae_d <= 0.5 * mae_z,
        "ap_intervals_separate": ci_ap_d[0] > ci_ap_z[1],
        "mae_intervals_separate": ci_mae_d[1] < ci_mae_z[0],
    }
    return {
        "rule": rule,
        **outcome(all(checks.values())),
        "checks": checks,
        "n_images": d["n_images"],
        "centre_ap": {
            "dfine": ap_d,
            "dfine_ci95": ci_ap_d,
            "zeroshot": ap_z,
            "zeroshot_ci95": ci_ap_z,
        },
        "count_mae": {
            "dfine": mae_d,
            "dfine_ci95": ci_mae_d,
            "zeroshot": mae_z,
            "zeroshot_ci95": ci_mae_z,
        },
    }


def h2(oof: dict) -> dict:
    rule = "Relative count MAE for images with 20+ animals >= 2 x that for images with 1-4."
    sparse, dense = band(oof["by_band"], 1), band(oof["by_band"], 20)
    if not sparse or not dense or "relative_mae" not in sparse or "relative_mae" not in dense:
        return {"rule": rule, **outcome(None, "a herd-size band has no images")}
    ratio = dense["relative_mae"] / sparse["relative_mae"] if sparse["relative_mae"] else None
    ok = dense["relative_mae"] >= 2 * sparse["relative_mae"]
    return {
        "rule": rule,
        **outcome(ok),
        "ratio": ratio,
        "band_1_to_4": sparse,
        "band_20_plus": dense,
        "intervals_separate": (
            None
            if not (sparse.get("relative_mae_ci95") and dense.get("relative_mae_ci95"))
            else dense["relative_mae_ci95"][0] > sparse["relative_mae_ci95"][1]
        ),
    }


def h3(oof: dict) -> dict:
    rule = "The nominal 90% count interval contains the true count on 85% to 95% of images."
    iv = oof["intervals"]["0.90"]
    return {
        "rule": rule,
        **outcome(0.85 <= iv["coverage"] <= 0.95),
        "coverage": iv["coverage"],
        "coverage_ci95": iv["coverage_ci95"],
        "coverage_by_band": iv["coverage_by_band"],
    }


def h4(oof: dict) -> dict:
    rule = "Recall at the operating point is at least 0.05 lower at 100 m than at 60 m."
    alt = oof["recall_by_altitude"]
    if "60m" not in alt or "100m" not in alt:
        return {"rule": rule, **outcome(None, "no images at 60 m or at 100 m")}
    gap = alt["60m"]["recall"] - alt["100m"]["recall"]
    return {"rule": rule, **outcome(gap >= 0.05), "recall_60m_minus_100m": gap, "by_altitude": alt}


def h5(oof: dict) -> dict:
    rule = "Fewer than 0.05 false detections per image on images with no cattle."
    empty = band(oof["by_band"], 0)
    if not empty or "false_positives_per_image" not in empty:
        return {"rule": rule, **outcome(None, "no images without cattle")}
    return {
        "rule": rule,
        **outcome(empty["false_positives_per_image"] < 0.05),
        "false_positives_per_image": empty["false_positives_per_image"],
        "ci95": empty["false_positives_per_image_ci95"],
        "n_images": empty["n"],
    }


def h6(s3: dict | None) -> dict:
    rule = (
        "The naive sum of per-image counts is more than 20% above the verified number of animals "
        "on every verified flight whose labelled images overlap."
    )
    v = (s3 or {}).get("verified")
    if not v or not v.get("h6_flights_with_overlap"):
        return {"rule": rule, **outcome(None, "no verified flight with overlapping images")}
    return {
        "rule": rule,
        **outcome(bool(v["h6_naive_over_20pct_on_every_overlapping_flight"])),
        "flights": v["h6_flights_with_overlap"],
        "naive_relative_error_min": v["h6_naive_relative_error_min"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=REPO_ROOT / "results" / "baselines")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    dfine = load(args.dir, "dfine_scores.json")
    if dfine is None:
        raise SystemExit(f"{args.dir / 'dfine_scores.json'} missing: run run_dfine_baselines.py")
    oof = dfine["out_of_fold"]
    result = {
        "H1": h1(dfine, load(args.dir, "zeroshot_scores.json")),
        "H2": h2(oof),
        "H3": h3(oof),
        "H4": h4(oof),
        "H5": h5(oof),
        "H6": h6(load(args.dir, "dfine_s3.json")),
    }
    dest = args.out or args.dir / "hypotheses.json"
    dest.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    for k, r in result.items():
        print(k, r["outcome"], r.get("reason", ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
