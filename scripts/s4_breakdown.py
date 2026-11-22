"""S4 broken down by farm size, herd density, loss type and each tricky case (Phase 4 plan).

Usage: python scripts/s4_breakdown.py [--farms N] [--workers N]

Uses the same simulated farms, error models and random streams as scripts/run_s4.py (validation
set only), so the overall figures here equal those in results/phase4/s4_validation.json. For the
naive rule and the interval, recount and farm rules at the 90% level, and the recount rule with the
large-loss setting swept, writes results/phase4/s4_breakdown_validation.json:
  * by farm size (animals on the farm);
  * by herd density (camps holding at least the dense-herd number of animals, and not);
  * by loss type and size (deaths and thefts; 1, 2-4, 5-9 and 10 or more animals);
  * false alarms by most likely cause (unrecorded moves, strays, register errors, partial surveys,
    dense herds, ordinary count error);
  * partial surveys against full ones;
  * across the range of each farm assumption in configs/sim_farms.yaml (lower, middle and upper
    third), so no single assumption decides an outcome.
Rates carry 95% bootstrap intervals over farms. Simulation results, never field results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from herdsight.herd.alerts import (
    FALSE_ALARM_CAUSES,
    Rule,
    false_alarm_causes,
    loss_outcomes,
    run_rule,
    score,
)
from herdsight.herd.sim import draw_settings, simulate_farm
from herdsight.utils.config import REPO_ROOT, load_config

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_s4 import LEVELS, WITHIN, boot, error_models  # noqa: E402

LEVEL = 0.9
BIG_SWEEP = (5.0, 10.0, 15.0, 20.0, float("inf"))
SIZE_BINS = ((20, 60), (60, 200), (200, 601))
LOSS_BINS = ((1, 1), (2, 4), (5, 9), (10, 10**6))
SWEPT = (
    "annual_mortality_range",
    "theft_share_per_year",
    "straying_events_per_camp_year",
    "unrecorded_move_share",
    "register_error_share",
    "surveys_per_month",
    "camps_per_farm_range",
)
SETTING_FIELD = {
    "annual_mortality_range": "annual_mortality",
    "theft_share_per_year": "theft_share_per_year",
    "straying_events_per_camp_year": "straying_per_camp_year",
    "unrecorded_move_share": "unrecorded_move_share",
    "register_error_share": "register_error_share",
    "surveys_per_month": "surveys_per_month",
    "camps_per_farm_range": "camps",
}


def rules() -> dict[str, Rule]:
    out = {
        "naive": Rule("naive"),
        f"interval@{LEVEL}": Rule("interval", level=LEVEL),
        f"recount@{LEVEL}": Rule("recount", level=LEVEL),
        f"farm@{LEVEL}": Rule("farm", level=LEVEL),
    }
    for big in BIG_SWEEP:
        if big != Rule("recount").big:
            out[f"recount@{LEVEL}_large{big:g}".replace("inf", "never")] = Rule(
                "recount", level=LEVEL, big=big
            )
    return out


def _bin(value: float, bins: tuple[tuple[int, int], ...], fmt: str) -> str:
    for lo, hi in bins:
        if lo <= value <= hi if fmt == "loss" else lo <= value < hi:
            if fmt == "loss":
                return f"{lo}" if lo == hi else (f"{lo}+" if hi > 10**5 else f"{lo}-{hi}")
            return f"{lo}-{hi - 1}" if hi < 601 else f"{lo}-600"
    raise ValueError(value)


def third(key: str, value: float, ranges: dict) -> str:
    lo, hi = ranges[key]
    if key in ("surveys_per_month", "camps_per_farm_range"):
        edges = np.linspace(lo, hi + 1, 4)
    else:
        edges = np.linspace(lo, hi, 4)
    return ("lower", "middle", "upper")[min(2, int(np.searchsorted(edges, value, "right") - 1))]


def farm_counts(args: tuple) -> list[dict[str, dict[str, float]]]:
    """Counters for one farm under every error model: {model: {rule: {key: value}}}."""
    i, s, seed, ranges = args
    models = error_models()
    rule_map = rules()
    out = {}
    for mi, m in enumerate(models):
        q = m.calibrate(LEVELS, np.random.default_rng(seed + 1))
        farm = simulate_farm(
            s,
            m,
            q,
            np.random.default_rng([seed, i]),
            obs_rng=np.random.default_rng([seed, i, mi]),
        )
        size = _bin(s.herd_size, SIZE_BINS, "size")
        groups = [f"size:{size}"] + [
            f"{k}:{third(k, getattr(s, SETTING_FIELD[k]), ranges)}" for k in SWEPT
        ]
        per_rule = {}
        for key, rule in rule_map.items():
            alerts = run_rule(farm, rule, m, q, np.random.default_rng([seed, i, 99]))
            c: Counter = Counter()
            k1 = score(farm, alerts, 1, WITHIN)
            c["surveys"] = k1["surveys"]
            c["fa"] = k1["false_alarms"]
            c["moved"] = k1["moved_notices"]
            for sv in farm.surveys:
                c[f"surveys|dense:{sv.dense}"] += 1
                c[f"surveys|partial:{sv.partial}"] += 1
            for a, cause in false_alarm_causes(farm, alerts):
                c[f"fa|cause:{cause}"] += 1
                c[f"fa|dense:{a['survey'].dense}"] += 1
                c[f"fa|partial:{a['survey'].partial}"] += 1
            for g in groups:
                c[f"surveys|{g}"] = k1["surveys"]
                c[f"fa|{g}"] = k1["false_alarms"]
            for lo in loss_outcomes(farm, alerts, WITHIN):
                tags = [
                    "all",
                    f"type:{lo['kind']}",
                    f"size:{_bin(lo['n'], LOSS_BINS, 'loss')}",
                    f"type:{lo['kind']}|size:{_bin(lo['n'], LOSS_BINS, 'loss')}",
                    f"camp_dense:{lo['camp_animals'] >= m.dense_at}",
                ]
                if lo["n"] >= 5:
                    tags += [f"k5|{g}" for g in groups] + [
                        "k5",
                        f"k5|camp_dense:{lo['camp_animals'] >= m.dense_at}",
                    ]
                for t in tags:
                    c[f"loss|{t}"] += 1
                    c[f"caught|{t}"] += lo["caught"]
                    if lo["caught"]:
                        c[f"delay|{t}"] += lo["surveys_until_caught"]
            per_rule[key] = dict(c)
        out[m.name] = per_rule
    return out


def summarise(farms: list[dict], model: str, key: str, rng: np.random.Generator) -> dict:
    rows = [f[model][key] for f in farms]
    keys = sorted({k for r in rows for k in r})

    def arr(k: str) -> np.ndarray:
        return np.array([r.get(k, 0) for r in rows], float)

    def rate(num: str, den: str, scale: float = 1.0) -> dict:
        n, d = arr(num), arr(den)
        ci = boot(n, d, rng)
        return {
            "value": float(scale * n.sum() / d.sum()) if d.sum() else None,
            "ci95": [scale * x for x in ci] if ci else None,
            "denominator": int(d.sum()),
        }

    out: dict = {
        "false_alarms_per_100_surveys": rate("fa", "surveys", 100),
        "false_alarms": int(arr("fa").sum()),
    }
    out["false_alarm_causes"] = {
        c: {
            "false_alarms": int(arr(f"fa|cause:{c}").sum()),
            "share": float(arr(f"fa|cause:{c}").sum() / max(arr("fa").sum(), 1)),
            "per_100_surveys": float(100 * arr(f"fa|cause:{c}").sum() / arr("surveys").sum()),
        }
        for c in FALSE_ALARM_CAUSES
    }
    out["by_survey"] = {
        f"{kind}:{flag}": rate(f"fa|{kind}:{flag}", f"surveys|{kind}:{flag}", 100)
        for kind in ("dense", "partial")
        for flag in (True, False)
    }
    groups = sorted({k.split("|", 1)[1] for k in keys if k.startswith("surveys|") and ":" in k})
    groups = [g for g in groups if not g.startswith(("dense:", "partial:"))]
    out["by_farm"] = {}
    for g in groups:
        out["by_farm"][g] = {
            "false_alarms_per_100_surveys": rate(f"fa|{g}", f"surveys|{g}", 100),
            "losses_5plus_caught_share": rate(f"caught|k5|{g}", f"loss|k5|{g}"),
        }
    tags = sorted({k.split("|", 1)[1] for k in keys if k.startswith("loss|")})
    out["losses"] = {}
    for t in tags:
        caught = arr(f"caught|{t}").sum()
        out["losses"][t] = {
            "caught_share": rate(f"caught|{t}", f"loss|{t}"),
            "mean_surveys_until_caught": float(arr(f"delay|{t}").sum() / caught)
            if caught
            else None,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--farms", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    cfg = load_config("sim_farms")
    ranges = {k: p.value for k, p in cfg.params.items()}
    n = args.farms or int(cfg.require("farms_validation"))
    seed = int(cfg.require("seed_validation"))
    settings_rng = np.random.default_rng(seed)
    farm_settings = [draw_settings(ranges, settings_rng) for _ in range(n)]
    jobs = [(i, s, seed, ranges) for i, s in enumerate(farm_settings)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        farms = list(pool.map(farm_counts, jobs, chunksize=20))
    models = [m.name for m in error_models()]
    results: dict = {
        "set": "validation",
        "farms": n,
        "seed": seed,
        "sim_farms_hash": cfg.hash(),
        "within_surveys": WITHIN,
        "level": LEVEL,
        "dense_herd_animals": error_models()[0].dense_at,
        "note": "Simulation results on simulated farms; not field results.",
        "runs": [],
    }
    for model in models:
        brng = np.random.default_rng(seed + 3)
        for key, rule in rules().items():
            big = None if np.isinf(rule.big) else rule.big  # None: no large-loss shortcut
            row = {"error_model": model, "rule": key, "large_loss": big}
            row.update(summarise(farms, model, key, brng))
            results["runs"].append(row)
            print(
                model,
                key,
                round(row["false_alarms_per_100_surveys"]["value"], 2),
                round(row["losses"]["k5"]["caught_share"]["value"], 3),
                flush=True,
            )
    out_dir = REPO_ROOT / "results" / "phase4"
    blob = json.dumps(results, indent=1, allow_nan=False, default=str) + "\n"
    dest = out_dir / (
        "s4_breakdown_validation.json" if args.farms is None else f"s4_breakdown_{n}farms.json"
    )
    dest.write_text(blob, encoding="utf-8")
    print("written", dest, hashlib.sha256(blob.encode()).hexdigest()[:12])
    return 0


if __name__ == "__main__":
    sys.exit(main())
