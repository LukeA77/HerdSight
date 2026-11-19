"""Write results/splits.json from the ICAERUS inventory, by the rules in herdsight.data.splits.

Usage: python scripts/make_splits.py
Run once, before any model is run; the file is committed so its date precedes every result.
Refuses to overwrite an existing splits file.
"""

from __future__ import annotations

import json
import sys

from herdsight.data.splits import FlightStats, make_splits
from herdsight.utils.config import REPO_ROOT

INVENTORY = REPO_ROOT / "results" / "data" / "icaerus_inventory.json"
OUT = REPO_ROOT / "results" / "splits.json"


def main() -> int:
    if OUT.exists():
        print(f"{OUT} exists; splits are fixed once. Delete it only with a logged decision.")
        return 1
    inv = json.loads(INVENTORY.read_text(encoding="utf-8"))
    flights = [
        FlightStats(f["farm"], f["flight"], f["n_labelled"], f["n_boxes"]) for f in inv["flights"]
    ]
    splits = {"dataset": inv["dataset"], "icaerus_v2": make_splits(flights)}
    splits["not_split_here"] = {
        "shao_2020": "second unseen-source test, evaluation only (D-21); not opened before Phase 7"
    }
    OUT.write_text(json.dumps(splits, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(splits["icaerus_v2"]["summary"], indent=1))
    print("test farm:", splits["icaerus_v2"]["test_farm"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
