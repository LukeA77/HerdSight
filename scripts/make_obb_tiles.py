"""Oriented labels for each fold's training tiles (Phase 2, step G7).

Usage: python scripts/make_obb_tiles.py --labels <labels.jsonl> --package <tile package> --out DIR

Applies the C6 rules to the SAM results (the recorded centre rule; flagged labels fall back to an
angle-0 box) and adds an "obb" to every annotation of train_fold<k>.json. Refuses to run unless the
SAM file is the recorded one and the eye check in results/phase2/c6_sam_labels.json has passed.
--allow-pending-review is for rehearsals only; no model from such a run may be kept.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from herdsight.labels.obb_tiles import attach_obbs, parse_tile_name
from herdsight.labels.sam_final import final_obb
from herdsight.utils.config import REPO_ROOT

LABELS_SHA256 = "61642f39d56cb019a6e665ac4f3a0996b04b6dcb6451a738fff04c8f776a62b1"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--allow-pending-review", action="store_true")
    args = ap.parse_args(argv)
    raw = args.labels.read_bytes()
    if hashlib.sha256(raw).hexdigest() != LABELS_SHA256:
        raise SystemExit("the SAM label file is not the recorded one")
    summary = json.loads((REPO_ROOT / "results/phase2/c6_sam_labels.json").read_text("utf-8"))
    if summary["eye_check"] != "passed" and not args.allow_pending_review:
        raise SystemExit(f"eye check is {summary['eye_check']}: oriented training stays paused")
    keep_centre = bool(summary["centres_kept_from_upright_labels"])
    sp = json.loads((REPO_ROOT / "results" / "splits.json").read_text())["icaerus_v2"]
    folds = [f"fold{k}" for k in range(sp["n_folds"])]
    by_image = {}
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["fold"] not in folds:
                raise SystemExit(f"image outside the folds: {row['image']}")
            farm, flight, name = row["image"].split("/")
            by_image[(farm, flight, name.rsplit(".", 1)[0])] = (
                row["fold"],
                row["boxes"],
                [
                    final_obb(b, x, keep_centre).tolist()
                    for b, x in zip(row["boxes"], row["labels"], strict=True)
                ],
            )
    args.out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for fold in folds:
        coco = json.loads((args.package / f"train_{fold}.json").read_text())
        anns: dict[int, list[dict]] = {}
        for a in coco["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)
        n = 0
        for im in coco["images"]:
            mine = anns.get(im["id"])
            if not mine:
                continue
            t = parse_tile_name(im["file_name"])
            source_fold, boxes, obbs = by_image[(t.farm, t.flight, t.stem)]
            if source_fold == fold:
                raise SystemExit(f"tile from the held-out fold: {im['file_name']}")
            out = attach_obbs(
                [a["bbox"] for a in mine], t.x0, t.y0, im["width"], im["height"], boxes, obbs
            )
            for a, o in zip(mine, out, strict=True):
                a["obb"] = [round(float(v), 4) for v in o]
                n += 1
        (args.out / f"train_{fold}.json").write_text(json.dumps(coco))
        counts[fold] = {"tiles": len(coco["images"]), "oriented_labels": n}
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
