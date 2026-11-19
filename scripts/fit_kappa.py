"""Turn the blind re-annotation answers into kappa for centre matching (brief 9.1).

Usage: python scripts/fit_kappa.py <centre_answers.json>

For each answered task, the annotator's box is matched to the nearest original box in the same
crop (by centre). With d the centre distance and sigma the original box's short side:

    kappa = sqrt(mean((d / sigma)^2))

Why: if each annotator places a centre with independent per-axis noise tau * sigma, the
difference between two annotators has per-axis variance 2 tau^2 sigma^2, so
mean(d^2 / sigma^2) = 4 tau^2. COCO's keypoint constant is k = 2 tau (Ronchi and Perona, 2017;
COCO keypoint evaluation), which gives kappa = sqrt(mean(d^2 / sigma^2)). A bootstrap interval
over tasks is reported with it.

Writes results/data/kappa_fit.json. The value is then copied, with that file as its source,
into configs/evaluation.yaml (centre_kappa).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from herdsight.utils.config import REPO_ROOT
from herdsight.utils.paths import storage_path


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    answers = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    key = json.loads((storage_path("exports", "kappa_task") / "key.json").read_text())
    ratios, skipped, unmatched = [], 0, 0
    for tid, a in sorted(answers.items()):
        if a.get("skipped") or "box" not in a:
            skipped += 1
            continue
        k = key[tid]
        boxes = np.asarray(k["boxes_display"], float).reshape(-1, 4)
        if boxes.size == 0:
            unmatched += 1
            continue
        b = np.asarray(a["box"], float)
        c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
        centres = np.column_stack(
            [(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2]
        )
        j = int(np.argmin(np.hypot(*(centres - c).T)))
        sigma = min(boxes[j, 2] - boxes[j, 0], boxes[j, 3] - boxes[j, 1])
        ratios.append(float(np.hypot(*(centres[j] - c)) / sigma))
    r = np.asarray(ratios)
    if r.size < 30:
        print(f"only {r.size} usable answers; at least 30 are needed")
        return 1
    rng = np.random.default_rng(0)
    boots = [np.sqrt(np.mean(rng.choice(r, r.size) ** 2)) for _ in range(2000)]
    result = {
        "kappa": float(np.sqrt(np.mean(r**2))),
        "kappa_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "n_answers": int(r.size),
        "n_skipped": skipped,
        "n_unmatched": unmatched,
        "normalised_distance_median": float(np.median(r)),
        "normalised_distance_p90": float(np.percentile(r, 90)),
        "rule": "kappa = sqrt(mean((d / short_side)^2)) between owner and original box centres",
    }
    dest = REPO_ROOT / "results" / "data" / "kappa_fit.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
