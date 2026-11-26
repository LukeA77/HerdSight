"""Label files for the farm onboarding curve (Phase 2 step G13, rule in the decisions log).

Usage: python gpu/make_onboarding_sets.py --package <tile package> --out <folder>
           --farm Jalogny --pool-days 20230623 20230627 --k 50 168 [--seed 0]

Models the offer to a farm: start from the farms the model already knows, add the labelled photos
of the farm's first survey days, and measure on its later days. From the union of the package's
train_fold0..2.json (as gpu/make_lofo_sets.py) it writes, into --out:
  train_onb_<farm>_k<k>.json  every tile of the other farms, plus every tile of k photos of <farm>
                              drawn (seeded) from the pool days;
  test_onb_<farm>.json        every tile of <farm> taken on any other day (never trained on);
  monitor_onb_<farm>_k<k>.json  up to 400 test tiles that show animals (logged only);
  onb_<farm>_photos.json      the photos drawn for each k, for the record.
A k larger than the pool takes the whole pool. The test farms are not in the package.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path

from make_lofo_sets import MONITOR_MAX_TILES, coco, union


def day_of(tile: str) -> str:
    m = re.search(r"DJI_(\d{8})", tile)
    if not m:
        raise SystemExit(f"no survey date in tile name {tile}")
    return m.group(1)


def photo_of(tile: str) -> str:
    return tile.rsplit("__", 1)[0]


def make_sets(
    package: Path, out: Path, farm: str, pool_days: list[str], ks: list[int], seed: int = 0
) -> dict:
    images, anns, categories = union(package)
    out.mkdir(parents=True, exist_ok=True)
    link = out / "images"
    if not link.exists():
        try:
            link.symlink_to((package / "images").resolve(), target_is_directory=True)
        except OSError:  # Windows without the symlink right: fine for a local check only
            print("no images link made (train on Linux, as on Kaggle)")
    shutil.copyfile(package / "manifest.json", out / "manifest.json")
    others = sorted(n for n in images if not n.startswith(f"{farm}__"))
    mine = [n for n in images if n.startswith(f"{farm}__")]
    pool = sorted({photo_of(n) for n in mine if day_of(n) in pool_days})
    test = sorted(n for n in mine if day_of(n) not in pool_days)
    if not pool or not test:
        raise SystemExit("the pool days and the test days must both have tiles")
    order = pool[:]
    random.Random(seed).shuffle(order)
    drawn, counts = {}, {"pool_photos": len(pool), "test_tiles": len(test)}
    for k in ks:
        chosen = set(order[: min(k, len(order))])  # nested: the k=50 photos are in k=200's
        tiles = sorted(n for n in mine if photo_of(n) in chosen)
        (out / f"train_onb_{farm}_k{k}.json").write_text(
            json.dumps(coco(others + tiles, images, anns, categories))
        )
        drawn[k] = sorted(chosen)
        counts[f"k{k}"] = {
            "photos": len(chosen),
            "farm_tiles": len(tiles),
            "all_tiles": len(others) + len(tiles),
        }
    (out / f"test_onb_{farm}.json").write_text(json.dumps(coco(test, images, anns, categories)))
    monitor = json.dumps(
        coco([n for n in test if anns[n]][:MONITOR_MAX_TILES], images, anns, categories)
    )
    for k in ks:  # gpu/train_dfine.py reads monitor_<fold>.json next to train_<fold>.json
        (out / f"monitor_onb_{farm}_k{k}.json").write_text(monitor)
    record = {"farm": farm, "pool_days": pool_days, "seed": seed, "photos": drawn, "counts": counts}
    (out / f"onb_{farm}_photos.json").write_text(json.dumps(record, indent=1))
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--farm", required=True)
    ap.add_argument("--pool-days", nargs="+", required=True)
    ap.add_argument("--k", type=int, nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(
        json.dumps(make_sets(args.package, args.out, args.farm, args.pool_days, args.k, args.seed))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
