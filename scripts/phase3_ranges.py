"""Phase 3 step 6 on real flights: count correction and conformal count ranges, out of fold.

Usage: python scripts/phase3_ranges.py   (after scripts/phase3_grouping.py)

Counting method: greedy joining with the new pair evidence ("greedy_fixed"), because global
grouping did not beat it (H23 not held; D-30's rule keeps the greedy stage); the Phase 1 method
and global grouping are reported beside it. Detections: the Phase 1 detector, out of fold.

For each fold, the calibration (herdsight.count.ranges) is fitted on the flights of the other two
folds and applied to its own flights:
* correction: the count times exp(median log(true / count)); kept only if it lowers the mean
  absolute relative error out of fold (plan step 6);
* ranges at 68, 90 and 95%: coverage (truth inside, whole animals) and mean relative width.
With 20 flights, real coverage is indicative (H26 is judged on simulated flights). Also writes the
calibration from all 20 flights, which simulated flights use (they are separate data). Writes
results/phase3/ranges_real.json.
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np

from herdsight.count.ranges import fit, inside
from herdsight.utils.config import REPO_ROOT, load_config
from herdsight.utils.env_report import env_report

METHODS = ("greedy_fixed", "phase1", "global_fixed")


def main() -> int:
    p3 = load_config("phase3")
    levels = tuple(float(x) for x in p3.require("interval_levels"))
    sp = json.loads((REPO_ROOT / p3.require("folds")).read_text())["icaerus_v2"]["flights"]
    g = json.loads((REPO_ROOT / "results" / "phase3" / "grouping_real.json").read_text())
    flights = g["dfine"]["flights"]
    keys = sorted(flights)
    fold = {k: sp[k] for k in keys}
    out: dict = {"levels": levels, "detections": "dfine (Phase 1, out of fold)", "methods": {}}
    for mth in METHODS:
        rows = []
        for k in keys:
            tr = [x for x in keys if fold[x] != fold[k]]
            cal = fit([flights[x][mth] for x in tr], [flights[x]["truth"] for x in tr], levels)
            c, t = flights[k][mth], flights[k]["truth"]
            row = {
                "flight": k,
                "count": c,
                "truth": t,
                "corrected": cal.corrected(c),
                "calibration_flights": cal.n,
            }
            for lv in levels:
                lo, hi = cal.interval(c, lv)
                row[f"{lv:.2f}"] = {
                    "lo": lo,
                    "hi": hi if math.isfinite(hi) else None,
                    "inside": inside(t, lo, hi),
                    "relative_width": (hi - lo) / t if math.isfinite(hi) else None,
                }
            rows.append(row)
        raw = np.mean([abs(r["count"] - r["truth"]) / r["truth"] for r in rows])
        cor = np.mean([abs(r["corrected"] - r["truth"]) / r["truth"] for r in rows])
        cov = {}
        for lv in levels:
            key = f"{lv:.2f}"
            widths = [
                r[key]["relative_width"] for r in rows if r[key]["relative_width"] is not None
            ]
            cov[key] = {
                "coverage": float(np.mean([r[key]["inside"] for r in rows])),
                "finite_ranges": len(widths),
                "mean_relative_width": float(np.mean(widths)) if widths else None,
            }
        full = fit([flights[x][mth] for x in keys], [flights[x]["truth"] for x in keys], levels)
        out["methods"][mth] = {
            "mare_raw": float(raw),
            "mare_corrected": float(cor),
            "correction_kept": bool(cor < raw),
            "coverage": cov,
            "calibration_all_flights": {
                "centre": full.centre,
                "q": {f"{lv:.2f}": (v if math.isfinite(v) else None) for lv, v in full.q.items()},
                "n": full.n,
            },
            "flights": rows,
        }
        print(
            f"{mth}: MARE raw {raw:.3f}, corrected {cor:.3f}; coverage "
            + ", ".join(
                f"{k} {v['coverage']:.2f} (width {v['mean_relative_width'] or float('nan'):.2f}, "
                f"{v['finite_ranges']} finite)"
                for k, v in cov.items()
            )
        )
    out["environment"] = env_report()
    (REPO_ROOT / "results" / "phase3" / "ranges_real.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
