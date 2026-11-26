"""Make the owner's blind count check of the S3 truth (Phase 3 plan, Risks: truth bias).

Usage: python scripts/make_blind_count.py            writes exports/blind_count/index.html
       python scripts/make_blind_count.py --score <answers.csv>

The S3 truth was built by correcting label-based candidate links, so it may lean towards that
linking. Here the owner counts whole flights from the photos alone: no boxes, no links, no counts.
Rule (fixed before the page was made, 25 Nov 2026): one flight per farm-date group, the one with
the fewest photos among the verified flights with at most 50 photos; at most 5 flights, fewest
photos first. Only photos with at least one labelled animal are shown (empty photos add time, not
information), in time order, reduced to a 1,600 px long side. --score compares the answers with
the verified counts (results/data/s3_truth.json) and writes results/phase3/blind_count.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from herdsight.data.icaerus import default_root, iter_images
from herdsight.utils.config import REPO_ROOT
from herdsight.utils.io import load_rgb
from herdsight.utils.paths import storage_path

MAX_PHOTOS, MAX_FLIGHTS, LONG_SIDE = 50, 5, 1600


def choose(truth: dict, root: Path) -> list[str]:
    groups = json.loads((REPO_ROOT / "results" / "splits.json").read_text())["icaerus_v2"]["groups"]
    best: dict[str, tuple[int, str]] = {}
    for key in sorted(truth):
        farm, flight = key.split("/")
        n = len(list((root / farm / "JPGImages" / flight).glob("*.JPG")))
        date = flight.split("_")[1][:8]
        g = f"{farm}/{date}"
        if g in groups and n <= MAX_PHOTOS and (g not in best or n < best[g][0]):
            best[g] = (n, key)
    return [key for _, key in sorted(best.values())][:MAX_FLIGHTS]


def make(out: Path) -> list[str]:
    truth = json.loads((REPO_ROOT / "results" / "data" / "s3_truth.json").read_text())["flights"]
    root = default_root()
    chosen = choose(truth, root)
    with_animals: dict[str, list[Path]] = {k: [] for k in chosen}
    for img, _problem in iter_images(root, with_metadata=False):
        key = f"{img.farm}/{img.flight}"
        if key in with_animals and img.boxes_xyxy is not None and len(img.boxes_xyxy):
            with_animals[key].append(Path(img.path))
    sections = []
    for i, key in enumerate(chosen, 1):
        folder = out / f"flight{i}"
        folder.mkdir(parents=True, exist_ok=True)
        pics = []
        for j, p in enumerate(sorted(with_animals[key]), 1):
            im = load_rgb(p)
            s = LONG_SIDE / max(im.size)
            im.resize((round(im.size[0] * s), round(im.size[1] * s))).save(
                folder / f"{j:02d}.jpg", quality=85
            )
            pics.append(f'<img src="flight{i}/{j:02d}.jpg" loading="lazy" alt="photo {j}">')
        sections.append(
            f'<section><h2>Flight {i} ({len(pics)} photos)</h2><div class="strip">{"".join(pics)}'
            f"</div><label>How many different animals are in flight {i}? "
            f'<input type="number" min="0" id="f{i}"></label></section>'
        )
    page = f"""<!doctype html><html lang="en-GB"><meta charset="utf-8">
<title>Blind count</title><style>body{{font:16px system-ui;margin:16px}}.strip{{display:flex;
gap:8px;overflow-x:auto}}.strip img{{height:420px}}section{{margin-bottom:32px}}input{{font:inherit;
width:6em}}</style><h1>Blind count</h1>
<p>Count the different animals in each flight. The photos overlap, so the same animal can appear in
several photos: count it once. Take your time; zoom in where needed. Please do not look at any
other material about these flights while counting.</p>{"".join(sections)}
<button onclick="dl()">Download answers</button>
<script>function dl(){{let c="flight,count\\n";for(let i=1;i<={len(chosen)};i++)c+=i+","+
document.getElementById("f"+i).value+"\\n";let a=document.createElement("a");a.href=
URL.createObjectURL(new Blob([c],{{type:"text/csv"}}));a.download="blind_count_answers.csv";
a.click()}}</script></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")
    (out / "flights.json").write_text(json.dumps(chosen, indent=1))  # kept away from the page
    return chosen


def score(answers: Path, out: Path) -> dict:
    truth = json.loads((REPO_ROOT / "results" / "data" / "s3_truth.json").read_text())["flights"]
    chosen = json.loads((out / "flights.json").read_text())
    rows = {int(r["flight"]): r["count"] for r in csv.DictReader(answers.open())}
    res = []
    for i, key in enumerate(chosen, 1):
        if not str(rows.get(i, "")).strip():
            continue
        blind, verified = int(rows[i]), int(truth[key]["verified_unique"])
        res.append(
            {
                "flight": key,
                "blind": blind,
                "verified": verified,
                "relative_difference": round((blind - verified) / verified, 4),
            }
        )
    mean_abs = sum(abs(r["relative_difference"]) for r in res) / max(len(res), 1)
    result = {
        "note": "Public data; one person's blind count of whole flights against the verified "
        "truth. Indicative only (few flights).",
        "rule": __doc__.split("Rule")[1].split("--score")[0].strip(),
        "flights": res,
        "mean_absolute_relative_difference": round(mean_abs, 4),
    }
    dest = REPO_ROOT / "results" / "phase3" / "blind_count.json"
    dest.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", type=Path, default=None)
    args = ap.parse_args(argv)
    out = storage_path("exports", "blind_count", create=True)
    if args.score:
        print(json.dumps(score(args.score, out), indent=1))
        return 0
    chosen = make(out)
    print(f"{len(chosen)} flights; open {out / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
