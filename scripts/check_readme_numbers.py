"""Check that the result numbers in the README come from results/readme_numbers.json.

Usage: python scripts/check_readme_numbers.py [README.md]

Every percentage and every decimal in the README's results sections ("Progress" to "How the work is
done") must match a value in results/readme_numbers.json (written by
scripts/make_readme_figures.py), as a percentage rounded to whole numbers or to one decimal
place, or as the value itself rounded to the same number of decimals. Numbers that are settings
or targets rather than results (the agreed 80%, the 15% target) are listed in ALLOWED with the
place they come from. Exits 1 and lists any number it cannot trace.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from herdsight.utils.config import REPO_ROOT

# Settings and targets quoted in the text, not results; each with where it is set.
ALLOWED = {
    "80%": "S1 targets and the onboarding rule (configs/specs.yaml, D-52)",
    "15%": "S3 target (configs/specs.yaml)",
    "65%": "S4 target (configs/specs.yaml)",
    "85%": "S2 range coverage target (configs/specs.yaml)",
    "10%": "thermal pass rule (docs/plans/phase5-simulation-mission.md)",
    "1.5": "S4 false-alarm target (configs/specs.yaml)",
    "0.45": "S2 target (configs/specs.yaml)",
    "5%": "text: 'within 5%' of the thermal count (results/phase5/thermal_count_study_v3.json)",
    "250%": "rounded from the add-up error (readme_numbers flight_count_error_add_up)",
    "70%": "status line: share of tracker tasks done (tracker)",
    "75%": "status line: share of tracker tasks done (tracker)",
}


def candidates(v: float) -> set[str]:
    out = {f"{v:g}", f"{v:.0f}", f"{v:.1f}", f"{v:.2f}"}
    if abs(v) <= 3:  # a share: also as a percentage
        out |= {f"{100 * v:.0f}%", f"{100 * v:.1f}%"}
    return out


def flatten(d: dict) -> list[float]:
    vals = []
    for v in d.values():
        if isinstance(v, bool):
            continue
        if isinstance(v, int | float):
            vals.append(float(v))
        elif isinstance(v, dict):
            vals += flatten(v)
        elif isinstance(v, list):
            vals += flatten(dict(enumerate(v)))
    return vals


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    readme = Path(args[0]) if args else REPO_ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    start, stop = text.find("## Progress"), text.find("## How the work is done")
    body = text[start:stop] if start >= 0 and stop > start else text
    nums = json.loads((REPO_ROOT / "results" / "readme_numbers.json").read_text(encoding="utf-8"))
    known: set[str] = set()
    for v in flatten(nums):
        known |= candidates(v)
    found = re.findall(r"(?<![\w.])(\d+(?:\.\d+)?%|\d+\.\d+)(?![\w.])", body)
    missing = sorted({x for x in found if x not in known and x not in ALLOWED})
    for x in missing:
        print("not traced to results/readme_numbers.json:", x)
    print(f"{len(found)} numbers checked, {len(missing)} not traced")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
