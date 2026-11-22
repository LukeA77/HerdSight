"""Phase 3 step 1: split the flight-count error into linking and detection (Phase 3 plan).

Usage:
  python scripts/score_s3.py --oracle --out results/phase3/oracle_s3_v20.json
  python scripts/score_s3.py --detector dfine --out results/phase3/dfine_s3_v20.json
  python scripts/phase3_error_split.py

Both inputs use the Phase 1 method (registration, then greedy grouping; D-27, D-30) on the
owner-verified flights (results/data/s3_truth.json, all 20 after review round 3). With the labels
as detections the error is the linking alone; with the Phase 1 detector's out-of-fold detections
it is detection and linking together; the difference per flight is what detection adds. Each
mean absolute relative error carries a 95% bootstrap interval over flights (resampled with a fixed
seed). Writes results/phase3/error_split.json.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from herdsight.utils.config import REPO_ROOT

N_BOOT = 5000
SEED = 0


def main() -> int:
    d = REPO_ROOT / "results" / "phase3"
    link = json.loads((d / "oracle_s3_v20.json").read_text())["flights"]
    full = json.loads((d / "dfine_s3_v20.json").read_text())["flights"]
    keys = sorted(k for k, v in link.items() if v["truth_kind"] == "owner-verified")
    if any(full[k]["truth_kind"] != "owner-verified" for k in keys):
        sys.exit("the two inputs were scored against different truth")
    e_link = np.array([link[k]["relative_error"]["registration"] for k in keys])
    e_full = np.array([full[k]["relative_error"]["registration"] for k in keys])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(keys), (N_BOOT, len(keys)))

    def summary(x: np.ndarray, absolute: bool = True) -> dict:
        v = np.abs(x) if absolute else x
        b = v[idx].mean(axis=1)
        return {
            "mean": float(v.mean()),
            "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
        }

    added = np.abs(e_full) - np.abs(e_link)
    farms = sorted({k.split("/")[0] for k in keys})
    out = {
        "flights": len(keys),
        "method": "Phase 1 registration and greedy grouping (D-27, D-30)",
        "linking_only_mare": summary(e_link),
        "linking_only_mean_signed": float(e_link.mean()),
        "detection_and_linking_mare": summary(e_full),
        "detection_and_linking_mean_signed": float(e_full.mean()),
        "added_by_detection": summary(added, absolute=False),
        "by_farm": {
            f: {
                "flights": int(sum(k.startswith(f + "/") for k in keys)),
                "linking_only_mare": float(
                    np.mean(
                        [
                            abs(link[k]["relative_error"]["registration"])
                            for k in keys
                            if k.startswith(f + "/")
                        ]
                    )
                ),
                "detection_and_linking_mare": float(
                    np.mean(
                        [
                            abs(full[k]["relative_error"]["registration"])
                            for k in keys
                            if k.startswith(f + "/")
                        ]
                    )
                ),
            }
            for f in farms
        },
        "per_flight": {
            k: {
                "fold": link[k]["fold"],
                "truth": link[k]["truth"],
                "overlapping_pairs": link[k]["overlapping_pairs"],
                "linking_only": float(link[k]["relative_error"]["registration"]),
                "detection_and_linking": float(full[k]["relative_error"]["registration"]),
                "labels_per_image_sum": link[k]["estimates"]["naive"],
                "detections_per_image_sum": full[k]["estimates"]["naive"],
            }
            for k in keys
        },
        "bootstrap": {"resamples": N_BOOT, "seed": SEED, "unit": "flight"},
    }
    (d / "error_split.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    lo, fu, ad = (
        out["linking_only_mare"],
        out["detection_and_linking_mare"],
        out["added_by_detection"],
    )
    print(
        f"{len(keys)} flights: linking only {lo['mean']:.3f} {np.round(lo['ci95'], 3)}; "
        f"detection and linking {fu['mean']:.3f} {np.round(fu['ci95'], 3)}; "
        f"added by detection {ad['mean']:.3f} {np.round(ad['ci95'], 3)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
