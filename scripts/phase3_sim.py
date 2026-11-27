"""Phase 3 on simulated validation flights: H23 (simulated part), H24 and H26.

Usage: python scripts/phase3_sim.py [--workers N] [--detector dfine|dfine_ref] [--set NAME]

Checks the Phase 3 freeze first. Flights: configs/phase3.yaml simulated_flights (count, on the
validation seed of configs/sim_worlds.yaml); each camp, herd and plan is drawn as in H20's moving
case (scripts/run_sim_flights.py), with the fitted detection, geotag and registration error
models. Each flight is counted five ways from one set of draws (herdsight.sim.phase3) with the
position-only pair models fitted on all 20 real verified flights.

* H23, simulated part: global grouping is not worse than greedy joining (fixed evidence): the
  upper end of the paired 95% interval of the difference in absolute relative error is at most
  the non-inferiority margin.
* H24: on flights with candidate pairs more than separate_pass_gap_s apart, global grouping with
  the movement allowance has a lower error than with the fixed gate, the paired interval
  excluding zero.
* H26: the counting method's 90% ranges, calibrated on the real flights only
  (results/phase3/ranges_real.json, greedy_fixed, all 20 flights), contain the true count on
  85-95% of simulated flights; 68% and 95% are reported too.
Writes results/phase3/sim_validation.json. With --detector dfine_ref (the final detector), the pair
models and ranges come from its own real-flight runs and the result is
sim_<set>_dfine_ref.json; --set test uses the test seed once, after scripts/freeze.py --phase
phase7_s3 (Phase 7). Simulation results, not field results.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

from herdsight.count.evidence import MotionModel, fit_logistic
from herdsight.count.ranges import Calibration, inside
from herdsight.mission.planner import PlannerSettings, plan_survey
from herdsight.sim.flight import build_world, fly
from herdsight.sim.herds import Camp, settings_from_config
from herdsight.sim.phase3 import POSITION_COLUMNS, PositionModels, phase3_counts
from herdsight.utils.config import REPO_ROOT, load_config
from herdsight.utils.env_report import env_report
from herdsight.utils.paths import storage_path
from herdsight.utils.run_once import claim_test_run

METHODS = ("phase1", "greedy_fixed", "global_fixed", "greedy_movement", "global_movement")
_MODELS: PositionModels | None = None
_SETTINGS: dict = {}


def fit_models(det_run: str = "dfine") -> PositionModels:
    d3 = storage_path("processed", "phase3")
    lab = np.load(d3 / "pairs_labels.npz")
    m = (lab["y"] == 1) & ~lab["geotag"]
    motion = MotionModel.fit(lab["d_m"][m], lab["dt_s"][m])
    det = np.load(d3 / f"pairs_{det_run}.npz")
    k = det["y"] >= 0
    fixed = fit_logistic(det["x_fixed"][k][:, POSITION_COLUMNS], det["y"][k])
    movement = fit_logistic(det["x_movement"][k][:, POSITION_COLUMNS], det["y"][k])
    return PositionModels(motion, fixed, movement)


def _init(models: PositionModels, settings: dict) -> None:
    global _MODELS, _SETTINGS
    _MODELS, _SETTINGS = models, settings


def one_flight(job: tuple[int, int]) -> dict:
    seed, i = job
    rng = np.random.default_rng([seed, i])
    camp = Camp.rectangle(rng.uniform(200, 600), rng.uniform(150, 450), f"camp{i}")
    n = int(rng.integers(20, 150))
    lo, hi = load_config("sim_worlds").require("drone_reaction_height_m")
    s = replace(settings_from_config(rng), reaction_height=(lo + hi) / 2)
    plan = plan_survey(camp, PlannerSettings.from_config("uk"))
    world = build_world(camp, n, np.random.default_rng([seed, i, 1]), herd_settings=s)
    flight = fly(world, plan, np.random.default_rng([seed, i, 2]))
    assert _MODELS is not None
    res = phase3_counts(
        flight,
        world.median_adult_m,
        np.random.default_rng([seed, i, 3]),
        _MODELS,
        _SETTINGS["gap_s"],
        _SETTINGS["max_exact"],
        _SETTINGS["time_limit_s"],
    )
    return {"camp_ha": camp.area_m2() / 1e4, "truth": flight.true_count, "animals": n, **res}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--detector", default="dfine")
    ap.add_argument("--set", default="validation", choices=("validation", "test"))
    args = ap.parse_args(argv)
    sfx = "" if args.detector == "dfine" else f"_{args.detector}"
    if args.set == "test" and not sfx:
        raise SystemExit("the test run uses the final detector (--detector dfine_ref)")
    dest = REPO_ROOT / "results" / "phase3" / f"sim_{args.set}{sfx}.json"
    if args.set == "test":  # one-time run on the test seed (brief section 9.5)
        claim_test_run("phase7_s3", dest)
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
    simc, boot, rules = (
        p3.require("simulated_flights"),
        p3.require("bootstrap"),
        p3.require("hypotheses"),
    )
    seed_key = simc["seed"] if args.set == "validation" else "seed_test"
    seed = int(load_config("sim_worlds").require(seed_key))
    ex = p3.require("exact_solver")
    settings = {
        "gap_s": float(p3.require("separate_pass_gap_s")),
        "max_exact": ex["max_detections"],
        "time_limit_s": ex["time_limit_s"],
    }
    models = fit_models(args.detector)
    jobs = [(seed, i) for i in range(int(simc["count"]))]
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(models, settings)) as pool:
        rows = list(pool.map(one_flight, jobs))
    rows = [r for r in rows if r["truth"] > 0]
    truth = np.array([r["truth"] for r in rows], float)
    err = {m: np.array([(r[m] - r["truth"]) for r in rows]) / truth for m in METHODS}
    rng = np.random.default_rng(boot["seed"])
    idx = rng.integers(0, len(rows), (boot["resamples"], len(rows)))

    def ci(v: np.ndarray, sub: np.ndarray | None = None) -> list[float]:
        v = v if sub is None else v[sub]
        ii = idx if sub is None else rng.integers(0, len(v), (boot["resamples"], len(v)))
        b = v[ii].mean(axis=1)
        return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]

    summary = {
        m: {"mare": float(np.abs(e).mean()), "mare_ci95": ci(np.abs(e)), "signed": float(e.mean())}
        for m, e in err.items()
    }
    d23 = np.abs(err["global_fixed"]) - np.abs(err["greedy_fixed"])
    h23 = {
        "global_minus_greedy": float(d23.mean()),
        "ci95": ci(d23),
        "margin": rules["H23"]["sim_noninferiority_margin"],
    }
    h23["not_worse"] = bool(h23["ci95"][1] <= h23["margin"])
    sub = np.array([r["separate_passes"] for r in rows])
    h24: dict = {"flights": int(sub.sum())}
    if sub.sum() >= 2:
        d24 = np.abs(err["global_movement"]) - np.abs(err["global_fixed"])
        h24 |= {"movement_minus_fixed": float(d24[sub].mean()), "ci95": ci(d24, sub)}
        h24["holds_in_simulation"] = bool(h24["movement_minus_fixed"] < 0 and h24["ci95"][1] < 0)
    real = json.loads((REPO_ROOT / "results" / "phase3" / f"ranges_real{sfx}.json").read_text())
    calj = real["methods"]["greedy_fixed"]["calibration_all_flights"]
    cal = Calibration(
        calj["centre"],
        {float(k): (v if v is not None else math.inf) for k, v in calj["q"].items()},
        calj["n"],
    )
    cov = {}
    for lv in cal.q:
        hits = [inside(r["truth"], *cal.interval(r["greedy_fixed"], lv)) for r in rows]
        cov[f"{lv:.2f}"] = float(np.mean(hits))
    r26 = rules["H26"]
    h26 = {
        "coverage_simulated": cov,
        "coverage_real_out_of_fold": {
            k: v["coverage"] for k, v in real["methods"]["greedy_fixed"]["coverage"].items()
        },
        "holds": bool(r26["coverage_90_min"] <= cov["0.90"] <= r26["coverage_90_max"]),
    }
    out = {
        "set": args.set,
        "detector": args.detector,
        "seed": seed,
        "flights": len(rows),
        "note": "Simulation results on simulated camps; not field results.",
        "position_models": {
            "diffusion_m2_s": models.motion.diffusion_m2_s,
            "fixed_coef": models.fixed.coef.tolist(),
            "movement_coef": models.movement.coef.tolist(),
        },
        "summary": summary,
        "H23_sim": h23,
        "H24_sim": h24,
        "H26": h26,
        "rows": rows,
        "environment": env_report(),
    }
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    for m, s in summary.items():
        print(
            f"{m:16s} MARE {s['mare']:.3f} {np.round(s['mare_ci95'], 3)} signed {s['signed']:+.3f}"
        )
    print("H23 sim:", h23)
    print("H24 sim:", h24)
    print("H26:", h26)
    return 0


if __name__ == "__main__":
    sys.exit(main())
