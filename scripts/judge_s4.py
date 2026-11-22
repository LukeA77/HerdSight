"""Judge Phase 4 hypotheses H13-H15 from results/phase4/s4_validation.json (400 farms) and
H16-H17 from results/phase4/s4_validation_4000farms.json.

Usage: python scripts/judge_s4.py

Rules as written in docs/hypotheses.md (Phase 4 section) before any alert rule was run: the
90% interval level and the count error model measured in Phase 1 with independent surveys
(`measured_rho0.0`). Writes results/phase4/hypotheses.json; outcomes are "held" or "not held".
"""

from __future__ import annotations

import json
import sys

from herdsight.utils.config import REPO_ROOT, load_config

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

    # second round (written after the breakdowns), on the 4000 validation farms
    big = json.loads(
        (REPO_ROOT / "results" / "phase4" / "s4_validation_4000farms.json").read_text()
    )
    runs4 = {r["rule"]: r for r in big["runs"] if r["error_model"] == MODEL}
    cur, strict = runs4[f"recount@{LEVEL}"], runs4[f"recount@{LEVEL}_shortcut0.99"]
    out["H16"] = {
        "rule": "with the large-loss shortcut on the 99% interval, recount false alarms at least "
        "40% lower and share of losses of 5+ caught at most 0.03 lower (4000 farms)",
        "farms": big["farms"],
        "current_fa": cur[fa],
        "strict_fa": strict[fa],
        "current_k5_caught": cur["k5_caught_share"],
        "strict_k5_caught": strict["k5_caught_share"],
    }
    out["H16"]["outcome"] = (
        "held"
        if strict[fa] <= 0.6 * cur[fa]
        and cur["k5_caught_share"] - strict["k5_caught_share"] <= 0.03
        else "not held"
    )
    target = load_config("specs").require("s4_alert_targets")
    levels = {}
    for key, r in runs4.items():
        if key.startswith("recount@") and key.endswith("_shortcut0.99"):
            lv = float(key.split("@")[1].split("_")[0])
            levels[lv] = {
                "fa": r[fa],
                "k5_caught": r["k5_caught_share"],
                "meets": r[fa] <= target["false_alarms_per_100_surveys_max"]
                and r["k5_caught_share"] >= target["losses_of_5_plus_caught_within_3_surveys_min"],
            }
    out["H17"] = {
        "rule": "recount with the stricter shortcut meets the S4 target at one or more levels",
        "target": target,
        "levels": {str(k): v for k, v in sorted(levels.items())},
        "outcome": "held" if any(v["meets"] for v in levels.values()) else "not held",
    }
    # third round: the history rule (Kalman filter of each camp's shortfall)
    hist = {}
    for key, r in runs4.items():
        if key.startswith("history@"):
            lv = float(key.split("@")[1])
            hist[lv] = {
                "fa": r[fa],
                "k5_caught": r["k5_caught_share"],
                "meets_target": r[fa] <= target["false_alarms_per_100_surveys_max"]
                and r["k5_caught_share"] >= target["losses_of_5_plus_caught_within_3_surveys_min"],
            }
    if hist:
        base = cur["k5_caught_share"]
        out["H18"] = {
            "rule": "history rule at some level: at most 1.5 false alarms per 100 surveys and "
            "share of losses of 5+ caught at least 0.05 above the recount rule at 90%",
            "recount_k5_caught": base,
            "levels": {str(k): v for k, v in sorted(hist.items())},
            "outcome": "held"
            if any(v["fa"] <= 1.5 and v["k5_caught"] >= base + 0.05 for v in hist.values())
            else "not held",
        }
        out["H19"] = {
            "rule": "history rule meets the S4 target at one or more levels",
            "outcome": "held" if any(v["meets_target"] for v in hist.values()) else "not held",
        }
    dest = REPO_ROOT / "results" / "phase4" / "hypotheses.json"
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
