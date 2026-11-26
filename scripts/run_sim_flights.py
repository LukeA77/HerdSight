"""Simulated survey flights without ROS: H20 and H21 (Phase 5 plan; validation seeds only).

Usage: python scripts/run_sim_flights.py [--experiment h20|h21|h27] [--flights N] [--workers N]

H21: camps whose animals stand in one or two packed groups of 20 or more, flown with and without
herd mode (same world and draws); H21 holds if the mean error with herd mode is at most two
thirds of that without it. Writes results/phase5/h21.json.

H20:

For each flight: a camp (size, herd size and herd settings drawn on the validation seed), the
planned survey, and the herd flown twice over the same world with the same detection, geotag and
registration draws: once with the animals still and once moving (measured grazing drift, walking
groups, reaction to the drone at the mid-range reaction height of configs/sim_worlds.yaml). The
flight error is |count - true| / true. H20 (docs/hypotheses.md): the mean error with movement is
at least 1.5 times the mean error with still animals. Paired 95% bootstrap interval of the ratio
over flights. Writes results/phase5/h20.json. Simulation results, never field results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

from herdsight.mission.planner import PlannerSettings, plan_survey
from herdsight.sim.flight import build_world, count_flight, fly
from herdsight.sim.herds import Camp, settings_from_config
from herdsight.utils.config import REPO_ROOT, load_config

RATIO_MIN = 1.5  # H20
H21_MAX = 2 / 3  # H21


def herd_mode_flight(job: tuple[int, int], second_look: bool = False) -> dict:
    seed, i = job
    rng = np.random.default_rng([seed, 21, i])
    camp = Camp.rectangle(rng.uniform(250, 500), rng.uniform(200, 400), f"camp{i}")
    n = int(rng.integers(40, 150))
    s = replace(settings_from_config(rng), groups=int(rng.integers(1, 3)))
    plan = plan_survey(camp, PlannerSettings.from_config("uk"))
    out = {"camp_ha": camp.area_m2() / 1e4, "animals": n, "groups": s.groups}
    arms = ("survey_only", "herd_mode") + (("second_look",) if second_look else ())
    for arm in arms:
        world = build_world(camp, n, np.random.default_rng([seed, 21, i, 1]), herd_settings=s)
        flight = fly(
            world,
            plan,
            np.random.default_rng([seed, 21, i, 2]),
            herd_mode=arm == "herd_mode",
            second_look=arm == "second_look",
        )
        res = count_flight(flight, world.median_adult_m, np.random.default_rng([seed, 21, i, 3]))
        out[arm] = {
            "count": res["count"],
            "error": abs(res["count"] - n) / n,
            "signed": (res["count"] - n) / n,
            "herd_mode_stills": res["herd_mode_stills"],
            "false_triggers": flight.false_triggers,
            "min_descent_clearance_m": flight.min_descent_clearance_m,
            "watched_spots": len(flight.watches),
        }
    return out


def second_look_flight(job: tuple[int, int]) -> dict:
    return herd_mode_flight(job, second_look=True)


def run_h27(rows: list[dict], seed: int) -> dict:
    a = np.array([r["survey_only"]["error"] for r in rows])
    h = np.array([r["herd_mode"]["error"] for r in rows])
    b = np.array([r["second_look"]["error"] for r in rows])
    rng = np.random.default_rng(seed + 27)
    boot = []
    for _ in range(2000):
        k = rng.integers(0, len(rows), len(rows))
        boot.append(b[k].mean() / max(a[k].mean(), 1e-9))
    ratio = float(b.mean() / max(a.mean(), 1e-9))
    held = ratio <= H21_MAX and b.mean() < h.mean()
    return {
        "mean_error_survey_only": float(a.mean()),
        "mean_error_herd_mode": float(h.mean()),
        "mean_error_second_look": float(b.mean()),
        "mean_signed_second_look": float(np.mean([r["second_look"]["signed"] for r in rows])),
        "ratio_second_look_to_survey": ratio,
        "ratio_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "watched_spots": int(sum(r["second_look"]["watched_spots"] for r in rows)),
        "H27_rule": "held if second-look / survey-only mean error <= 2/3 and below herd mode",
        "H27": "held" if held else "not held",
    }


def run_h21(rows: list[dict], seed: int) -> dict:
    a = np.array([r["survey_only"]["error"] for r in rows])
    b = np.array([r["herd_mode"]["error"] for r in rows])
    rng = np.random.default_rng(seed + 21)
    boot = []
    for _ in range(2000):
        k = rng.integers(0, len(rows), len(rows))
        boot.append(b[k].mean() / max(a[k].mean(), 1e-9))
    ratio = float(b.mean() / max(a.mean(), 1e-9))
    return {
        "mean_error_survey_only": float(a.mean()),
        "mean_error_herd_mode": float(b.mean()),
        "mean_signed_survey_only": float(np.mean([r["survey_only"]["signed"] for r in rows])),
        "mean_signed_herd_mode": float(np.mean([r["herd_mode"]["signed"] for r in rows])),
        "ratio": ratio,
        "ratio_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "herd_mode_stills": int(sum(r["herd_mode"]["herd_mode_stills"] for r in rows)),
        "H21_rule": "held if herd-mode / survey-only mean error <= 2/3",
        "H21": "held" if ratio <= H21_MAX else "not held",
    }


def one_flight(job: tuple[int, int]) -> dict:
    seed, i = job
    rng = np.random.default_rng([seed, i])
    width, height = rng.uniform(200, 600), rng.uniform(150, 450)
    camp = Camp.rectangle(width, height, f"camp{i}")
    n = int(rng.integers(20, 150))
    lo, hi = load_config("sim_worlds").require("drone_reaction_height_m")
    s = replace(settings_from_config(rng), reaction_height=(lo + hi) / 2)
    plan = plan_survey(camp, PlannerSettings.from_config("uk"))
    out = {"camp_ha": camp.area_m2() / 1e4, "animals": n, "photos": plan.n_photos}
    for still in (True, False):
        hs = replace(s, grazing_diffusion=0.0, walking_share=0.0) if still else s
        wrng = np.random.default_rng([seed, i, 1])  # same world and draws for both runs
        world = build_world(camp, n, wrng, herd_settings=hs)
        if still:
            world.herd.state = ["rest"] * len(world.herd.state)
            world.herd.bout_left = [1e12] * len(world.herd.bout_left)
        flight = fly(world, plan, np.random.default_rng([seed, i, 2]))
        res = count_flight(flight, world.median_adult_m, np.random.default_rng([seed, i, 3]))
        key = "still" if still else "moving"
        out[key] = {
            "count": res["count"],
            "error": abs(res["count"] - n) / n,
            "signed": (res["count"] - n) / n,
            "reacted": flight.reacted,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="h20", choices=("h20", "h21", "h27"))
    ap.add_argument("--flights", type=int, default=60)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    seed = int(load_config("sim_worlds").require("seed_validation"))
    if args.experiment in ("h21", "h27"):
        h27 = args.experiment == "h27"
        fn = second_look_flight if h27 else herd_mode_flight
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(pool.map(fn, [(seed, i) for i in range(args.flights)]))
        out = {
            "set": "validation",
            "seed": seed,
            "flights": len(rows),
            "note": "Simulation results on simulated camps; not field results.",
            **(run_h27(rows, seed) if h27 else run_h21(rows, seed)),
            "rows": rows,
        }
        dest = REPO_ROOT / "results" / "phase5"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / ("h27.json" if h27 else "h21.json")).write_text(
            json.dumps(out, indent=1) + "\n", encoding="utf-8"
        )
        print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
        return 0
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(one_flight, [(seed, i) for i in range(args.flights)]))
    still = np.array([r["still"]["error"] for r in rows])
    moving = np.array([r["moving"]["error"] for r in rows])
    rng = np.random.default_rng(seed + 9)
    boot = []
    for _ in range(2000):
        k = rng.integers(0, len(rows), len(rows))
        boot.append(moving[k].mean() / max(still[k].mean(), 1e-9))
    ratio = float(moving.mean() / max(still.mean(), 1e-9))
    out = {
        "set": "validation",
        "seed": seed,
        "flights": len(rows),
        "note": "Simulation results on simulated camps; not field results.",
        "mean_error_still": float(still.mean()),
        "mean_error_moving": float(moving.mean()),
        "mean_signed_error_still": float(np.mean([r["still"]["signed"] for r in rows])),
        "mean_signed_error_moving": float(np.mean([r["moving"]["signed"] for r in rows])),
        "ratio": ratio,
        "ratio_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "animals_that_reacted": int(sum(r["moving"]["reacted"] for r in rows)),
        "H20_rule": f"held if moving / still mean error >= {RATIO_MIN}",
        "H20": "held" if ratio >= RATIO_MIN else "not held",
        "rows": rows,
    }
    dest = REPO_ROOT / "results" / "phase5"
    dest.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(out, indent=1) + "\n"
    (dest / "h20.json").write_text(blob, encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    print("hash", hashlib.sha256(blob.encode()).hexdigest()[:12])
    return 0


if __name__ == "__main__":
    sys.exit(main())
