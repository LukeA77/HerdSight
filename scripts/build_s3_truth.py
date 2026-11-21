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

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from herdsight.utils.config import REPO_ROOT
from herdsight.utils.paths import storage_path


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    answers: dict = {}
    review: dict = {"tasks": []}
    for arg in argv:
        path = Path(arg)
        answers |= json.loads(path.read_text(encoding="utf-8"))
        review["tasks"] += json.loads((path.parent / "tasks.json").read_text())["tasks"]
    links_dir = storage_path("processed", "s3_links")
    summary = json.loads((REPO_ROOT / "results" / "data" / "s3_candidate_links.json").read_text())
    by_flight: dict[str, list] = defaultdict(list)
    for t in review["tasks"]:
        by_flight[t["flight"]].append(t)
    out = {}
    for key, tasks in sorted(by_flight.items()):
        farm, flight = key.split("/")
        det = json.loads((links_dir / f"{farm}__{flight}.json").read_text())
        owner = {}
        comps = [list(map(tuple, c)) for c in det["components"]]
        for ci, c in enumerate(comps):
            for m in c:
                owner[m] = ci
        splits = merges = missing_labels = unanswered = 0
        for t in tasks:
            a = answers.get(t["id"])
            if a is None:
                unanswered += 1
                continue
            if t["kind"] == "same":
                for n in a.get("different", []):
                    m = tuple(t["pics"][n]["member"])
                    ci = owner[m]
                    comps[ci].remove(m)
                    comps.append([m])
                    owner[m] = len(comps) - 1
                    splits += 1
            elif a.get("answer") == "boxed" and t["other"][1] is not None:
                ca, cb = owner[tuple(t["member"])], owner[tuple(t["other"])]
                if ca != cb:
                    for m in comps[cb]:
                        owner[m] = ca
                    comps[ca] += comps[cb]
                    comps[cb] = []
                    merges += 1
            elif a.get("answer") == "unboxed":
                missing_labels += 1
        comps = [c for c in comps if c]
        conflicts = sum(1 for c in comps if len({i for i, _ in c}) < len(c))
        verified = sum(max(Counter(i for i, _ in c).values()) for c in comps)
        out[key] = {
            "naive_sum": summary[key]["naive_sum"],
            "candidate_unique": summary[key]["candidate_unique"],
            "verified_unique": verified,
            "verified_groups_before_conflict_rule": len(comps),
            "splits": splits,
            "merges": merges,
            "missing_labels_found": missing_labels,
            "remaining_conflicts": conflicts,
            "tasks": len(tasks),
            "unanswered": unanswered,
            "complete": unanswered == 0,
        }
    dest = REPO_ROOT / "results" / "data" / "s3_truth.json"
    dest.write_text(
        json.dumps({"flights": out, "rules": __doc__.split("Rules:")[1].strip()}, indent=1) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
