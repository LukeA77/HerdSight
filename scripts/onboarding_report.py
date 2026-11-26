"""Plain-language farm onboarding report from the onboarding curve (G13 / Stage B step 6).

Usage: python scripts/onboarding_report.py [--farm Jalogny] [--out <file.html>]

Reads results/phase2/g13_onboarding.json (scripts/g13_curve.py) and writes, per farm, the short
report a farmer would receive after the free set-up: how many of their photos were used, how much
of their cattle the system found on later survey days before and after learning their farm, and
whether the agreed level was reached. Wording follows the dashboard's rules (no technical terms).
On public data this is a demonstration of the report, not a result for a real farm.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

from herdsight.herd.messages import banned_words
from herdsight.utils.config import REPO_ROOT
from herdsight.utils.paths import storage_path

CURVE = REPO_ROOT / "results" / "phase2" / "g13_onboarding.json"


def farm_text(farm: str, d: dict) -> list[str]:
    curve = {int(k): v for k, v in d["curve"].items()}
    base = curve.get(0)
    best_k = max(k for k in curve if k > 0) if any(k > 0 for k in curve) else None
    counts = (d.get("setup") or {}).get("counts", {})

    def used(k: int) -> int:  # photos actually used (a farm's pool can be smaller than k)
        return int(counts.get(f"k{k}", {}).get("photos", k))

    lines = [f"Report for {farm}"]
    if base:
        lines.append(
            f"Before learning your farm, the system found {base['centre_recall']:.0%} of your "
            f"cattle in photos from later checks."
        )
    for k in sorted(k for k in curve if k > 0):
        v = curve[k]
        lines.append(
            f"After learning from {used(k)} photos of your farm, it found "
            f"{v['centre_recall']:.0%} of your cattle, and {v['centre_precision']:.0%} of what it "
            f"marked were really cattle."
        )
    if d["supported"]:
        lines.append(
            f"The agreed level (at least 80% found, at least 80% correct) was reached with "
            f"{used(d['fewest_photos_meeting_rule'])} photos of your farm."
        )
    elif best_k is not None:
        lines.append(
            "The agreed level (at least 80% found, at least 80% correct) was not reached. Under "
            "the offer you do not pay, and your photos are deleted."
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--farm", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if not CURVE.exists():
        sys.exit("run scripts/g13_curve.py first")
    data = json.loads(CURVE.read_text(encoding="utf-8"))["farms"]
    farms = [args.farm] if args.farm else sorted(data)
    sections = []
    for farm in farms:
        lines = farm_text(farm, data[farm])
        for line in lines:
            if banned_words(line):
                sys.exit(f"technical words in the report: {banned_words(line)}")
        body = "".join(f"<p>{html.escape(x)}</p>" for x in lines[1:])
        sections.append(f"<section><h2>{html.escape(lines[0])}</h2>{body}</section>")
        print("\n".join(lines) + "\n")
    page = (
        '<!doctype html><html lang="en-GB"><meta charset="utf-8"><title>Onboarding report</title>'
        "<style>body{font:17px/1.5 system-ui;max-width:680px;margin:24px auto;padding:0 16px}"
        "</style><p><em>Demonstration on public data (ICAERUS), not a real customer farm.</em></p>"
        + "".join(sections)
        + "</html>"
    )
    out = args.out or storage_path("exports", create=True) / "onboarding_report.html"
    out.write_text(page, encoding="utf-8")
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
