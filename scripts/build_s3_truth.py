"""Apply the owner's link review to the candidate identities: the S3 ground truth (brief 8.3).

Usage: python scripts/build_s3_truth.py <answers.json> [<answers.json> ...]

Each answers file must sit in its review folder, next to that review's tasks.json (one file per
review round; scripts/make_link_review.py).

Rules:
* "Same animal?": a picture marked as a different animal is split off as its own animal. This is
  cautious: two split-off pictures could be one animal, so the count can only err upwards, and
  the number of splits is reported.
* "Missed link?": "boxed" joins the lone animal to the nearest box at the matching place in the
  other image; "unboxed" is recorded as a missing label (the count is unchanged, since the animal
  is already counted once); "absent" and "unsure" change nothing.
* Unanswered tasks are reported, and the flight is marked incomplete.
* A "boxed" answer can join two candidate animals that both appear in one image (a conflict).
  Such a group is at least as many animals as its most-repeated image shows, so the verified
  count adds that number for it, not 1. The number of conflicts is reported.

Writes results/data/s3_truth.json.
"""

# The review rules above are applied by herdsight.count.truth.apply_review.

from __future__ import annotations

import json
import sys
from pathlib import Path

from herdsight.count.truth import apply_review, load_review
from herdsight.utils.config import REPO_ROOT
from herdsight.utils.paths import storage_path


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    answers, tasks, reviewed_flights = load_review([Path(a) for a in argv])
    links_dir = storage_path("processed", "s3_links")
    summary = json.loads((REPO_ROOT / "results" / "data" / "s3_candidate_links.json").read_text())
    out = {}
    for key in sorted(reviewed_flights):
        farm, flight = key.split("/")
        det = json.loads((links_dir / f"{farm}__{flight}.json").read_text())
        r = apply_review(det["components"], tasks[key], answers)
        out[key] = {
            "naive_sum": summary[key]["naive_sum"],
            "candidate_unique": summary[key]["candidate_unique"],
            "verified_unique": r.count,
            "verified_groups_before_conflict_rule": len(r.groups),
            "splits": r.splits,
            "merges": r.merges,
            "missing_labels_found": r.missing_labels,
            "remaining_conflicts": len(r.conflicts),
            "tasks": len(tasks[key]),
            "unanswered": r.unanswered,
            "complete": r.unanswered == 0,
        }
    dest = REPO_ROOT / "results" / "data" / "s3_truth.json"
    dest.write_text(
        json.dumps({"flights": out, "rules": __doc__.split("Rules:")[1].strip()}, indent=1) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
