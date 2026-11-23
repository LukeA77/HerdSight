"""Oriented-box labels for every cross-validation animal, with SAM (Phase 2, step C6).

Usage (with PyTorch and transformers): python scripts/make_sam_labels.py [--device cuda] [--limit N]

Writes $HERDSIGHT_STORAGE/runs/labels/sam_obb_v1/labels.jsonl, one line per image with its upright
boxes and each box's SAM result. A run that stops part-way resumes where it left off. Test-farm
images are never read. The labels are for training only and are never used to score a model.
Nothing is corrected here: acceptance, the centre rule and the fallback for flagged labels are
applied afterwards, as Phase 2 step C6 sets out.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from herdsight.data.icaerus import default_root, iter_images
from herdsight.labels.sam_obb import ObbLabel, SamObbLabeller
from herdsight.utils.config import REPO_ROOT, load_config
from herdsight.utils.env_report import env_report
from herdsight.utils.io import load_rgb
from herdsight.utils.paths import storage_path


def label_record(label: ObbLabel) -> dict[str, Any]:
    return {
        "obb": None if label.obb is None else [round(float(v), 4) for v in label.obb],
        "sam_score": round(label.sam_score, 4),
        "hull_iou": round(label.hull_iou_with_box, 4),
        "fill": round(label.mask_fill, 4),
        "flags": list(label.flags),
    }


def done_images(path: Path) -> set[str]:
    """Images already written; a line cut short by an interruption does not count."""
    done: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["image"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many images (a test)")
    ap.add_argument("--name", default="sam_obb_v1")
    args = ap.parse_args(argv)
    cfg = load_config("sam_labels")
    sp = json.loads((REPO_ROOT / "results" / "splits.json").read_text())["icaerus_v2"]
    folds = {f"fold{k}" for k in range(sp["n_folds"])}  # the split file also lists test flights
    images = []
    for s, problem in iter_images(default_root(), with_metadata=False):
        fold = sp["flights"].get(f"{s.farm}/{s.flight}")
        if fold in folds and problem is None and len(s.boxes_xyxy):
            images.append((f"{s.farm}/{s.flight}/{s.path.name}", fold, s))
    images.sort(key=lambda row: row[0])
    run_dir = storage_path("runs", "labels", args.name, create=True)
    path = run_dir / "labels.jsonl"
    done = done_images(path)
    todo = [row for row in images if row[0] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(images)} images with animals, {len(todo)} to do", flush=True)
    labeller = SamObbLabeller(
        cfg.require("sam_model"),
        cfg.require("sam_revision"),
        context=float(cfg.require("sam_crop_context")),
        escape_iou=float(cfg.require("sam_escape_iou")),
        min_fill=float(cfg.require("sam_min_fill")),
        device=args.device,
    )
    t0 = time.time()
    n_boxes = 0
    with path.open("a", encoding="utf-8") as fh:
        for i, (key, fold, s) in enumerate(todo, 1):
            im = load_rgb(s.path)
            boxes = s.boxes_xyxy
            labels = [label_record(labeller.label(im, b)) for b in boxes]
            row = {
                "image": key,
                "fold": fold,
                "boxes": [[round(float(v), 2) for v in b] for b in boxes],
                "labels": labels,
            }
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            n_boxes += len(boxes)
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  {(time.time() - t0) / n_boxes:.2f} s/box", flush=True)
    record = {
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": cfg.require("sam_model"),
        "revision": cfg.require("sam_revision"),
        "device": args.device,
        "images_with_animals": len(images),
        "images_written": len(done_images(path)),
        "labels": str(path),
        "environment": env_report(),
    }
    (run_dir / "run.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
