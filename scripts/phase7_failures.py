"""Where the final detector missed cattle on the held-back test farm (after the one-time run).

Usage: python scripts/phase7_failures.py

Reads the stored home-base predictions of the single Phase 7 run
($HERDSIGHT_STORAGE/runs/phase7/home_base.jsonl) and the threshold recorded with it
(results/phase7/test_farm.json); nothing is predicted, tuned or scored again. Each labelled animal
is matched with the project's centre matching and its share found is broken down by:
* crowding: another labelled animal within one body length (the cross-validation error model's
  measure, scripts/fit_detection_model.py);
* body length in pixels (thirds of the test farm's animals);
* animals in the photo (1-4, 5-19, 20 or more).
Also the share of false marks lying within one body length of a labelled animal (split or doubled
animals) rather than on empty ground. Writes results/phase7/test_farm_failures.json. Explains the
result; changes nothing. Public data, not a field result.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from herdsight.data.icaerus import default_root, iter_images
from herdsight.evaluate.centre_match import box_centres, match_hungarian, short_side
from herdsight.utils.config import REPO_ROOT, load_config
from herdsight.utils.paths import storage_path

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from score_baseline import load_preds  # noqa: E402


def main() -> int:
    res = json.loads((REPO_ROOT / "results" / "phase7" / "test_farm.json").read_text())
    thr = float(res["calibration"]["home_base"]["threshold"])
    ev = load_config("evaluation")
    kappa, sim = float(ev.require("centre_kappa")), float(ev.require("centre_match_similarity"))
    preds = load_preds(storage_path("runs", "phase7") / "home_base.jsonl")
    rows, false_near, false_far = [], 0, 0
    for s, problem in iter_images(default_root(), with_metadata=False):
        k = f"{s.farm}/{s.flight}/{s.path.name}"
        if s.farm != res["test_farm"] or problem is not None or k not in preds:
            continue
        t = np.asarray(s.boxes_xyxy, float).reshape(-1, 4)
        b, sc = preds[k]
        b = b[sc >= thr]
        found = np.zeros(len(t), bool)
        matched_pred = np.zeros(len(b), bool)
        if len(t) and len(b):
            m = match_hungarian(
                box_centres(b), box_centres(t), np.maximum(short_side(t), 1.0), kappa, sim
            )
            found[m.pairs[:, 1]] = True
            matched_pred[m.pairs[:, 0]] = True
        length = np.maximum(t[:, 2] - t[:, 0], t[:, 3] - t[:, 1]) if len(t) else np.zeros(0)
        tc = box_centres(t) if len(t) else np.zeros((0, 2))
        for i in range(len(t)):
            d = np.hypot(*(tc - tc[i]).T)
            d[i] = np.inf
            rows.append(
                {
                    "found": bool(found[i]),
                    "crowded": bool((d < length[i]).any()),
                    "length_px": float(length[i]),
                    "in_photo": len(t),
                }
            )
        if len(b):
            pc = box_centres(b)[~matched_pred]
            for c in pc:
                near = len(t) and (np.hypot(*(tc - c).T) < length).any()
                false_near += int(bool(near))
                false_far += int(not near)
    f = np.array([r["found"] for r in rows])
    crowd = np.array([r["crowded"] for r in rows])
    ln = np.array([r["length_px"] for r in rows])
    n_in = np.array([r["in_photo"] for r in rows])
    cuts = np.quantile(ln, [1 / 3, 2 / 3])

    def share(mask: np.ndarray) -> dict:
        return {"animals": int(mask.sum()), "found": float(f[mask].mean()) if mask.any() else None}

    out = {
        "note": (
            "Breakdown of the single Phase 7 run's stored predictions; nothing re-run or tuned. "
            "Public data, not a field result."
        ),
        "threshold": thr,
        "animals": len(rows),
        "found": float(f.mean()),
        "by_crowding": {"crowded": share(crowd), "apart": share(~crowd)},
        "by_length_px": {
            "cut_points_px": [float(x) for x in cuts],
            "small": share(ln < cuts[0]),
            "middle": share((ln >= cuts[0]) & (ln < cuts[1])),
            "large": share(ln >= cuts[1]),
        },
        "by_animals_in_photo": {
            "1-4": share(n_in < 5),
            "5-19": share((n_in >= 5) & (n_in < 20)),
            "20+": share(n_in >= 20),
        },
        "false_marks": {
            "near_an_animal": false_near,
            "on_empty_ground": false_far,
            "near_share": false_near / max(false_near + false_far, 1),
        },
    }
    dest = REPO_ROOT / "results" / "phase7" / "test_farm_failures.json"
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
