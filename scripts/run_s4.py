"""S4 baseline: alert rules on simulated farms (Phase 4 plan; D-13; H13-H15).

Usage: python scripts/run_s4.py [--farms N] [--set validation|test] [--tag NAME]

Simulates farms from configs/sim_farms.yaml (validation seed unless --set test, which is for the
Phase 7 run only), surveys each farm under several count error models, applies each alert rule at
each dial setting, and writes results/phase4/s4_<set>.json:
  * false alarms per 100 surveys;
  * share of losses (deaths, thefts) of at least 1 and at least 5 animals caught within 3 surveys;
  * mean surveys until caught;
each with a 95% bootstrap interval over farms. Then judges H13-H15 (validation only).

Count error models:
  measured  bias and spread of flight totals after duplicate removal on the owner-verified
            flights (results/baselines/dfine_s3.json, registration method), Phase 1 detector;
  target    unbiased, with the mean absolute error of the frozen S3 target
            (configs/specs.yaml s3_flight_error_max);
each with surveys of a camp independent (correlation 0) and correlated (0.5).
Simulation results, never field results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

import numpy as np

from herdsight.herd.alerts import Rule, run_rule, score
from herdsight.herd.sim import ErrorModel, draw_settings, simulate_farm
from herdsight.utils.config import REPO_ROOT, load_config

LEVELS = (0.68, 0.8, 0.9, 0.95, 0.99)
WITHIN = 3
N_BOOT = 1000


def error_models() -> list[ErrorModel]:
    s3 = json.loads((REPO_ROOT / "results" / "baselines" / "dfine_s3.json").read_text())
    rel = np.array(
        [
            f["relative_error"]["registration"]
            for f in s3["flights"].values()
            if f["truth_kind"] == "owner-verified"
        ]
    )
    target = float(load_config("specs").require("s3_flight_error_max"))
    sd_target = target / np.sqrt(2 / np.pi)  # mean |N(0, sd)| = target
    out = []
    for rho in (0.0, 0.5):
        out.append(
            ErrorModel(
                bias=float(rel.mean()),
                sd=float(rel.std(ddof=1)),
                correlation=rho,
                name=f"measured_rho{rho}",
            )
        )
        out.append(
            ErrorModel(bias=0.0, sd=float(sd_target), correlation=rho, name=f"target_rho{rho}")
        )
    return out


def rules() -> list[Rule]:
    out = [Rule("naive")]
    for name in ("interval", "recount", "farm"):
        out += [Rule(name, level=lv) for lv in LEVELS]
    return out


def boot(values: np.ndarray, weights: np.ndarray, rng: np.random.Generator) -> list[float] | None:
    """95% interval of sum(values) / sum(weights), resampling farms."""
    if weights.sum() == 0:
        return None
    n = len(values)
    stats = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        w = weights[i].sum()
        if w:
            stats.append(values[i].sum() / w)
    return [float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="validation", choices=("validation", "test"))
    ap.add_argument("--farms", type=int, default=None)
    ap.add_argument("--tag", default=None, help="save as s4_<set>_<tag>.json")
    args = ap.parse_args(argv)
    cfg = load_config("sim_farms")
    ranges = {k: p.value for k, p in cfg.params.items()}
    n = args.farms or int(cfg.require(f"farms_{args.set}"))
    seed = int(cfg.require(f"seed_{args.set}"))
    settings_rng = np.random.default_rng(seed)
    farm_settings = [draw_settings(ranges, settings_rng) for _ in range(n)]
    models = error_models()
    rule_list = rules()
    results: dict = {
        "set": args.set,
        "farms": n,
        "seed": seed,
        "sim_farms_hash": cfg.hash(),
        "within_surveys": WITHIN,
        "note": "Simulation results on simulated farms; not field results.",
        "error_models": [m.__dict__ for m in models],
        "runs": [],
    }
    for m in models:
        q = m.calibrate(LEVELS, np.random.default_rng(seed + 1))
        per_rule: dict[str, list[dict]] = {}
        unrecorded = []
        for i, s in enumerate(farm_settings):
            # the same farm history for every error model; survey errors differ by model
            farm = simulate_farm(
                s,
                m,
                q,
                np.random.default_rng([seed, i]),
                obs_rng=np.random.default_rng([seed, i, models.index(m)]),
            )
            for rule in rule_list:
                alerts = run_rule(farm, rule, m, q, np.random.default_rng([seed, i, 99]))
                key = f"{rule.name}@{rule.level}" if rule.name != "naive" else "naive"
                k1 = score(farm, alerts, 1, WITHIN)
                k5 = score(farm, alerts, 5, WITHIN)
                per_rule.setdefault(key, []).append(
                    {
                        **k1,
                        "k5_losses": k5["losses"],
                        "k5_caught": k5["caught"],
                        "k5_surveys_until_caught": k5["surveys_until_caught"],
                    }
                )
            unrecorded.append(s.unrecorded_move_share)
        brng = np.random.default_rng(seed + 2)
        for key, rows in per_rule.items():
            surveys = np.array([r["surveys"] for r in rows], float)
            fa = np.array([r["false_alarms"] for r in rows], float)
            row = {
                "error_model": m.name,
                "rule": key,
                "false_alarms_per_100_surveys": float(100 * fa.sum() / surveys.sum()),
                "false_alarms_per_100_surveys_ci95": [100 * x for x in boot(fa, surveys, brng)],
                "moved_notices_per_100_surveys": float(
                    100 * sum(r["moved_notices"] for r in rows) / surveys.sum()
                ),
            }
            for tag, lk, ck, dk in (
                ("k1", "losses", "caught", "surveys_until_caught"),
                ("k5", "k5_losses", "k5_caught", "k5_surveys_until_caught"),
            ):
                losses = np.array([r[lk] for r in rows], float)
                caught = np.array([r[ck] for r in rows], float)
                delays = [d for r in rows for d in r[dk]]
                row[f"{tag}_losses"] = int(losses.sum())
                row[f"{tag}_caught_share"] = float(caught.sum() / max(losses.sum(), 1))
                row[f"{tag}_caught_share_ci95"] = boot(caught, losses, brng)
                row[f"{tag}_mean_surveys_until_caught"] = float(np.mean(delays)) if delays else None
            moved = np.array(unrecorded) > 0.05
            row["false_alarms_per_100_surveys_farms_unrecorded_moves"] = float(
                100 * fa[moved].sum() / max(surveys[moved].sum(), 1)
            )
            results["runs"].append(row)
            print(
                m.name,
                key,
                round(row["false_alarms_per_100_surveys"], 2),
                round(row["k5_caught_share"], 3),
                flush=True,
            )
    out_dir = REPO_ROOT / "results" / "phase4"
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(results, indent=1) + "\n"
    results_hash = hashlib.sha256(blob.encode()).hexdigest()[:12]
    dest = out_dir / (f"s4_{args.set}_{args.tag}.json" if args.tag else f"s4_{args.set}.json")
    dest.write_text(blob, encoding="utf-8")
    print("written", dest, results_hash)
    return 0


if __name__ == "__main__":
    sys.exit(main())
