"""Write a simulated farm for one Gazebo run: the world, the herd state and the flight plan.

Usage: python sim/bridge/make_world.py --out DIR [--width 250] [--height 180] [--animals 30]
       [--seed N] [--profile uk]

Writes DIR/herdsight_farm.sdf (python/herdsight/sim/gazebo.py), DIR/herd_state.pkl (the herd the
world was made from, for sim/bridge/herd_bridge.py) and DIR/plan.yaml (the survey plan in PX4's
frame, python/herdsight/mission/planner.py: plan_to_dict). Simulation only.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from herdsight.mission.planner import PlannerSettings, plan_survey, plan_to_dict  # noqa: E402
from herdsight.sim.flight import build_world  # noqa: E402
from herdsight.sim.gazebo import world_sdf  # noqa: E402
from herdsight.sim.herds import Camp  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--width", type=float, default=250.0)
    ap.add_argument("--height", type=float, default=180.0)
    ap.add_argument("--animals", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", default="uk")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    # the camp starts beside the take-off point at the origin
    camp = Camp(
        np.array(
            [
                [20, -args.height / 2],
                [20 + args.width, -args.height / 2],
                [20 + args.width, args.height / 2],
                [20, args.height / 2],
            ],
            float,
        ),
        "Sim camp",
    )
    world = build_world(camp, args.animals, rng)
    (args.out / "herdsight_farm.sdf").write_text(world_sdf(world, seed=args.seed), encoding="utf-8")
    with (args.out / "herd_state.pkl").open("wb") as fh:
        pickle.dump(world.herd, fh)
    plan = plan_survey(camp, PlannerSettings.from_config(args.profile), launch=np.zeros(2))
    (args.out / "plan.yaml").write_text(yaml.safe_dump(plan_to_dict(plan)), encoding="utf-8")
    print(
        f"{world.herd.n} animals, {plan.n_photos} stills at {plan.height_agl:.0f} m -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
