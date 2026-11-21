"""Judge Phase 4 hypotheses H13-H15 from results/phase4/s4_validation.json.

Usage: python scripts/judge_s4.py

Rules as written in docs/hypotheses.md (Phase 4 section) before any alert rule was run: the
90% interval level and the count error model measured in Phase 1 with independent surveys
(`measured_rho0.0`). Writes results/phase4/hypotheses.json; outcomes are "held" or "not held".
"""

from __future__ import annotations

import json
import sys

from herdsight.utils.config import REPO_ROOT

MODEL = "measured_rho0.0"
LEVEL = 0.9


def main() -> int:
    d = json.loads((REPO_ROOT / "results" / "phase4" / "s4_validation.json").read_text())
    if d["set"] != "validation":
        raise SystemExit("hypotheses are judged on the validation set only")
    runs = {r["rule"]: r for r in d["runs"] if r["error_model"] == MODEL}
    naive, interval = runs["naive"], runs[f"interval@{LEVEL}"]
    recount, farm = runs[f"recount@{LEVEL}"], runs[f"farm@{LEVEL}"]
    fa = "false_alarms_per_100_surveys"
    fa_moves = "false_alarms_per_100_surveys_farms_unrecorded_moves"

    def delay(r: dict) -> float:
        return r["k5_mean_surveys_until_caught"] or float("inf")

    out = {
        "error_model": MODEL,
        "level": LEVEL,
        "note": "Simulation results on simulated farms; not field results.",
        "H13": {
            "rule": "interval false alarms at most 1/5 of naive, and share of losses of 5+ caught "
            "at most 0.10 lower",
            "naive_fa": naive[fa],
            "interval_fa": interval[fa],
            "naive_k5_caught": naive["k5_caught_share"],
            "interval_k5_caught": interval["k5_caught_share"],
        },
        "H14": {
            "rule": "recount false alarms at most half the interval rule's, and mean surveys "
            "until a loss of 5+ is caught at most 1 higher",
            "interval_fa": interval[fa],
            "recount_fa": recount[fa],
            "interval_delay": interval["k5_mean_surveys_until_caught"],
            "recount_delay": recount["k5_mean_surveys_until_caught"],
        },
        "H15": {
            "rule": "on farms with unrecorded-move share above 0.05, farm-rule false alarms at "
            "most half the recount rule's",
            "recount_fa": recount[fa_moves],
            "farm_fa": farm[fa_moves],
        },
    }
    out["H13"]["outcome"] = (
        "held"
        if interval[fa] * 5 <= naive[fa]
        and naive["k5_caught_share"] - interval["k5_caught_share"] <= 0.10
        else "not held"
    )
    out["H14"]["outcome"] = (
        "held"
        if recount[fa] <= 0.5 * interval[fa] and delay(recount) - delay(interval) <= 1
        else "not held"
    )
    out["H15"]["outcome"] = "held" if farm[fa_moves] <= 0.5 * recount[fa_moves] else "not held"
    dest = REPO_ROOT / "results" / "phase4" / "hypotheses.json"
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
