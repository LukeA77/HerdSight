"""Label files for leave-one-farm-out training (Phase 2 plan step G6, D-33).

Usage: python gpu/make_lofo_sets.py --package <tile package> --out <folder>

The tile package holds train_fold0..2.json; every cross-validation tile is in two of them. Their
union is split by farm. For each farm F this writes, into --out:
  train_lofo_F.json    every tile of the other two farms;
  monitor_lofo_F.json  up to 400 tiles of F that show animals, in name order (a progress check
                       only: the last checkpoint is always kept, nothing is selected on it).
The test farms are not in the package. An `images` link to the package's tiles is added, so --out
can be passed to gpu/train_dfine.py as the package with --fold lofo_F. The package manifest is
copied too, so each run records which package it was trained from.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

FARMS = ("Derval", "Jalogny", "Mauron")
MONITOR_MAX_TILES = 400


def union(package: Path) -> tuple[dict[str, dict], dict[str, list[dict]], list[dict]]:
    images: dict[str, dict] = {}
    anns: dict[str, list[dict]] = {}
    categories: list[dict] = []
    for k in range(3):
        c = json.loads((package / f"train_fold{k}.json").read_text())
        categories = c["categories"]
        name_of = {im["id"]: im["file_name"] for im in c["images"]}
        by_image: dict[str, list[dict]] = {name: [] for name in name_of.values()}
        for a in c["annotations"]:
            by_image[name_of[a["image_id"]]].append(a)
        for im in c["images"]:
            name = im["file_name"]
            if name in images:
                if len(anns[name]) != len(by_image[name]):
                    raise SystemExit(f"tile {name} has different labels in two folds")
                continue
            images[name], anns[name] = im, by_image[name]
    return images, anns, categories


def coco(names: list[str], images: dict, anns: dict, categories: list[dict]) -> dict:
    out_images, out_anns = [], []
    for name in names:
        new_id = len(out_images) + 1
        out_images.append({**images[name], "id": new_id})
        for a in anns[name]:
            out_anns.append({**a, "id": len(out_anns) + 1, "image_id": new_id})
    return {"images": out_images, "annotations": out_anns, "categories": categories}


def make_sets(package: Path, out: Path) -> dict[str, dict[str, int]]:
    images, anns, categories = union(package)
    farm_of = {name: name.split("__")[0] for name in images}
    unknown = set(farm_of.values()) - set(FARMS)
    if unknown:
        raise SystemExit(f"unexpected farms in the package: {sorted(unknown)}")
    out.mkdir(parents=True, exist_ok=True)
    link = out / "images"
    if not link.exists():
        link.symlink_to((package / "images").resolve(), target_is_directory=True)
    shutil.copyfile(package / "manifest.json", out / "manifest.json")  # recorded in run.json
    counts = {}
    for farm in FARMS:
        train = sorted(n for n in images if farm_of[n] != farm)
        monitor = sorted(n for n in images if farm_of[n] == farm and anns[n])[:MONITOR_MAX_TILES]
        (out / f"train_lofo_{farm}.json").write_text(
            json.dumps(coco(train, images, anns, categories))
        )
        (out / f"monitor_lofo_{farm}.json").write_text(
            json.dumps(coco(monitor, images, anns, categories))
        )
        counts[farm] = {"train_tiles": len(train), "monitor_tiles": len(monitor)}
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(make_sets(args.package, args.out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
