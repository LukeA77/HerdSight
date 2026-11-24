"""Per-tile animal ovals for one training set (dark-coat, tight-group and outline training).

Usage: python gpu/make_tile_ovals.py --labels <c6 labels.jsonl> --coco <train json> --out <json>

Reads the C6 rotated boxes (one row per photo, results/phase2/c6_sam_labels.json describes them)
and the training set's tile names, and writes {tile file name: [[cx, cy, length, width, angle],
...]} in tile pixels (gpu/tile_ovals.py). Only tiles of that training set are looked up, so for a
leave-one-farm-out set the held-out farm's animals are never read. Prints how many training
tiles with animals got ovals, and refuses to write if fewer than 95% did (a sign that the labels
and the tiles do not match).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tile_ovals import photo_key, tile_ovals  # noqa: E402

MIN_SHARE = 0.95


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--coco", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tile-px", type=int, default=640)
    args = ap.parse_args(argv)
    obbs: dict[str, list[list[float]]] = {}
    for line in args.labels.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            obbs[photo_key(row["image"])] = [lab["obb"] for lab in row["labels"]]
    coco = json.loads(args.coco.read_text())
    with_animals = {a["image_id"] for a in coco["annotations"]}
    names = [
        im["file_name"]
        for im in coco["images"]
        if im["id"] in with_animals and not im["file_name"].startswith("hardneg/")
    ]
    ovals = tile_ovals(obbs, names, args.tile_px)
    share = len(ovals) / max(len(names), 1)
    print(f"tiles with animals: {len(names)}; with ovals: {len(ovals)} ({share:.1%})")
    if share < MIN_SHARE:
        print(f"fewer than {MIN_SHARE:.0%} of tiles with animals have ovals; not written")
        return 1
    args.out.write_text(json.dumps(ovals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
