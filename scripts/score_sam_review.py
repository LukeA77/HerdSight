"""Score the eye check of 100 oriented labels (Phase 2, step C6).

Usage: python scripts/score_sam_review.py --answers <sam_review_answers.json>

Compares the answers with $HERDSIGHT_STORAGE/exports/sam_review/review_key.json. Only "good" counts
as accepted, against the frozen sam_eye_check_accept_min. Updates results/phase2/c6_sam_labels.json;
the G7 job refuses to train unless its eye_check reads "passed".
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from herdsight.labels.sam_final import review_outcome
from herdsight.utils.config import REPO_ROOT, load_config
from herdsight.utils.paths import storage_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", type=Path, required=True)
    args = ap.parse_args(argv)
    chk = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/freeze.py"),
            "--phase",
            "phase2_prereg",
            "--check",
        ],
        capture_output=True,
        text=True,
    )
    print(chk.stdout.strip())
    if chk.returncode != 0:
        return 1
    answers = json.loads(args.answers.read_text(encoding="utf-8"))
    key_path = storage_path("exports") / "sam_review" / "review_key.json"
    key = json.loads(key_path.read_text(encoding="utf-8"))
    accept_min = float(load_config("phase2").require("sam_eye_check_accept_min"))
    outcome = review_outcome(answers, key, accept_min)
    path = REPO_ROOT / "results" / "phase2" / "c6_sam_labels.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["eye_check"] = "passed" if outcome["passed"] else "failed"
    summary["eye_check_result"] = outcome
    path.write_bytes((json.dumps(summary, indent=1) + "\n").encode("utf-8"))
    print(json.dumps(outcome, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
