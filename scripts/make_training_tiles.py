"""Cut full-resolution training tiles for the Kaggle baseline (gpu/train_dfine_baselines_kaggle.py).

Usage: python scripts/make_training_tiles.py            (home-base detector, native resolution)
       python scripts/make_training_tiles.py --onboard  (onboard herd detector, S7: each image is
                                                         first reduced to onboard_long_side_px)

Reads results/splits.json (folds, D-26) and configs/baselines.yaml. Writes to the storage folder:
  datasets/processed/herdsight_tiles_v1/images/*.jpg        640 px tiles, native resolution
  datasets/processed/herdsight_tiles_v1/train_fold<k>.json  COCO labels: tiles from every fold
                                                            except k (the model for fold k)
  datasets/processed/herdsight_tiles_v1/monitor_fold<k>.json  up to 400 animal tiles of fold k,
                                                            logged during training, never used to
                                                            choose a checkpoint
  datasets/processed/herdsight_tiles_v1/manifest.json      counts, settings, splits hash
  exports/herdsight_tiles_v1.zip                           the package to upload to Kaggle

Every tile with at least one animal is kept; for each fold's training set, empty tiles are drawn
at random (seeded) to `empty_tile_ratio` times its number of animal tiles. Test-farm images are
never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from herdsight.data.icaerus import default_root, iter_images
from herdsight.detect.tiling import boxes_in_tile, tile_grid
from herdsight.utils.config import REPO_ROOT, load_config
from herdsight.utils.io import load_rgb
from herdsight.utils.paths import storage_path

MONITOR_MAX_TILES = 400


def coco_set(tiles: list[dict]) -> dict:
    images, anns = [], []
    for i, t in enumerate(tiles, 1):
        images.append({"id": i, "file_name": t["file"], "width": t["w"], "height": t["h"]})
        for x0, y0, x1, y1 in t["boxes"]:
            anns.append(
                {
                    "id": len(anns) + 1,
                    "image_id": i,
                    "category_id": 1,
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "area": (x1 - x0) * (y1 - y0),
                    "iscrowd": 0,
                }
            )
    return {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "cattle"}]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onboard", action="store_true")
    ap.add_argument(
        "--long-side",
        type=int,
        default=None,
        help="onboard only: reduce stills to this long side instead of onboard_long_side_px "
        "(package herdsight_onboard_tiles_<px>_v1; G10b, D-51)",
    )
    args = ap.parse_args(argv)
    if args.long_side and not args.onboard:
        ap.error("--long-side applies with --onboard only")
    cfg = load_config("baselines")
    tile = int(cfg.require("dfine_tile_px"))
    if args.onboard:
        NAME = "herdsight_onboard_tiles_v1"
        overlap = int(cfg.require("onboard_tile_overlap_px"))
        long_side = int(cfg.require("onboard_long_side_px"))
        if args.long_side:
            long_side = args.long_side
            NAME = f"herdsight_onboard_tiles_{long_side}_v1"
    else:
        NAME = "herdsight_tiles_v1"
        overlap = int(cfg.require("tile_overlap_px"))
        long_side = None
    ratio = float(cfg.require("empty_tile_ratio"))
    min_visible = float(cfg.require("min_visible_fraction"))
    rng = np.random.default_rng(int(cfg.require("sample_seed")))
    splits_path = REPO_ROOT / "results" / "splits.json"
    sp = json.loads(splits_path.read_text(encoding="utf-8"))["icaerus_v2"]
    folds = [f"fold{k}" for k in range(sp["n_folds"])]

    out = storage_path("processed", NAME, create=True)
    (out / "images").mkdir(exist_ok=True)
    positive: dict[str, list[dict]] = {f: [] for f in folds}
    empty: dict[str, list[tuple]] = {f: [] for f in folds}
    for s, problem in iter_images(default_root(), with_metadata=False):
        fold = sp["flights"].get(f"{s.farm}/{s.flight}")
        if fold not in folds or problem or not s.labelled:
            continue
        im = load_rgb(s.path)
        scale = 1.0
        if long_side:
            scale = long_side / max(im.size)
            im = im.resize(
                (round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS
            )
        for t in tile_grid(im.width, im.height, tile, overlap):
            boxes, _ = boxes_in_tile(t, s.boxes_xyxy * scale, min_visible)
            name = f"{s.farm}__{s.flight}__{s.path.stem}__{t.x0}_{t.y0}.jpg"
            # Animal tiles are saved now; empty tiles only when sampled later, except at onboard
            # scale, where tiles are small and reloading a resized image would cost more.
            if len(boxes) or long_side:
                im.crop((t.x0, t.y0, t.x1, t.y1)).save(out / "images" / name, quality=95)
            if len(boxes):
                positive[fold].append(
                    {
                        "file": name,
                        "w": t.x1 - t.x0,
                        "h": t.y1 - t.y0,
                        "boxes": [[round(float(v), 2) for v in b] for b in boxes],
                    }
                )
            else:
                empty[fold].append((s.path, (t.x0, t.y0, t.x1, t.y1), name))

    counts = {}
    for k in folds:
        train_pos = [t for f in folds if f != k for t in positive[f]]
        pool = [e for f in folds if f != k for e in empty[f]]
        n_empty = min(len(pool), int(round(ratio * len(train_pos))))
        picks = sorted(rng.choice(len(pool), size=n_empty, replace=False).tolist())
        train_empty = []
        for i in picks:
            path, box, name = pool[i]
            if not (out / "images" / name).exists():
                load_rgb(path).crop(box).save(out / "images" / name, quality=95)
            train_empty.append(
                {"file": name, "w": box[2] - box[0], "h": box[3] - box[1], "boxes": []}
            )
        train = coco_set(train_pos + train_empty)
        monitor = coco_set(positive[k][:MONITOR_MAX_TILES])
        (out / f"train_{k}.json").write_text(json.dumps(train), encoding="utf-8")
        (out / f"monitor_{k}.json").write_text(json.dumps(monitor), encoding="utf-8")
        counts[k] = {
            "train_tiles": len(train["images"]),
            "train_boxes": len(train["annotations"]),
            "train_empty_tiles": len(train_empty),
            "monitor_tiles": len(monitor["images"]),
        }
    manifest = {
        "name": NAME,
        "tile_px": tile,
        "overlap_px": overlap,
        "image_long_side_px": long_side or "native",
        "empty_tile_ratio": ratio,
        "min_visible_fraction": min_visible,
        "splits_sha256": hashlib.sha256(splits_path.read_bytes()).hexdigest(),
        "folds": folds,
        "counts": counts,
        "licence": "Tiles of ICAERUS v2 images, CC BY 4.0, Institut de l'Elevage (Zenodo 11048412)",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    keep = {p.name for p in (out / "images").iterdir()}
    used = set()
    for k in folds:
        for part in ("train", "monitor"):
            used |= {
                i["file_name"] for i in json.loads((out / f"{part}_{k}.json").read_text())["images"]
            }
    for name in keep - used:  # tiles cut on an earlier run and no longer referenced
        (out / "images" / name).unlink()
    zpath = storage_path("exports", create=True) / f"{NAME}.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                z.write(f, Path(NAME) / f.relative_to(out))
    manifest["zip_bytes"] = zpath.stat().st_size
    h = hashlib.sha256()
    with zpath.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    manifest["zip_sha256"] = h.hexdigest()
    dest = REPO_ROOT / "results" / "baselines" / f"{NAME}_manifest.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=1))
    print("upload", zpath)
    return 0


if __name__ == "__main__":
    sys.exit(main())
