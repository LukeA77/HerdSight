"""Out-of-fold predictions on degraded images for the resolution study (Phase 2, step G11).

Usage: python scripts/resolution_predict.py --models <folder> --kind g2 [--gpu]

<folder>/<kind>_fold<k>/last.onnx is the model trained without fold k. Each cross-validation image
is degraded at every scale in configs/resolution_study.yaml and predicted by that model. Output:
$HERDSIGHT_STORAGE/runs/phase2/g11/scale_<scale>/<fold>.jsonl, with boxes in full-image pixels. A
run that stops part-way resumes where it left off. Test-farm images are never read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from herdsight.data.icaerus import default_root, iter_images
from herdsight.detect.dfine_onnx import DfineOnnxTileDetector
from herdsight.detect.run_tiled import detect_full_image
from herdsight.evaluate.resolution import degrade, image_seed
from herdsight.utils.config import REPO_ROOT, load_config
from herdsight.utils.io import load_rgb
from herdsight.utils.paths import storage_path


def done_images(path: Path) -> set[str]:
    done: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["image"])
            except (json.JSONDecodeError, KeyError):
                pass  # a line cut short by an interruption
    return done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--kind", default="g2")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="images per fold (a test)")
    args = ap.parse_args(argv)
    cfg, base = load_config("resolution_study"), load_config("baselines")
    scales = [float(x) for x in cfg.require("scales")]
    blur, noise = float(cfg.require("blur_sigma_px")), float(cfg.require("noise_sd_8bit"))
    quality, seed = int(cfg.require("jpeg_quality")), int(cfg.require("seed"))
    tile, overlap = int(base.require("dfine_tile_px")), int(base.require("tile_overlap_px"))
    sp = json.loads((REPO_ROOT / "results" / "splits.json").read_text())["icaerus_v2"]
    folds = [f"fold{k}" for k in range(sp["n_folds"])]  # the split file also lists test flights
    by_fold: dict[str, list] = {f: [] for f in folds}
    for s, problem in iter_images(default_root(), with_metadata=False):
        fold = sp["flights"].get(f"{s.farm}/{s.flight}")
        if fold in by_fold and problem is None and s.labelled:
            by_fold[fold].append((f"{s.farm}/{s.flight}/{s.path.name}", s.path))
    root = storage_path("runs", "phase2", "g11", create=True)
    t0 = time.time()
    for fold in folds:
        rows = sorted(by_fold[fold], key=lambda r: r[0])
        if args.limit:
            rows = rows[: args.limit]
        detector = DfineOnnxTileDetector(
            str(args.models / f"{args.kind}_{fold}" / "last.onnx"),
            min_score=0.01,
            size=tile,
            gpu=args.gpu,
        )
        for scale in scales:
            out_dir = root / f"scale_{scale:.2f}"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{fold}.jsonl"
            done = done_images(path)
            todo = [r for r in rows if r[0] not in done]
            print(f"{fold} scale {scale:.2f}: {len(todo)} of {len(rows)} to do", flush=True)
            with path.open("a", encoding="utf-8") as fh:
                for key, image_path in todo:
                    im = load_rgb(image_path)
                    small = degrade(im, scale, blur, noise, quality, image_seed(key, scale, seed))
                    boxes, scores = detect_full_image(small, detector, tile, overlap)
                    factor = np.array(
                        [
                            im.width / small.width,
                            im.height / small.height,
                            im.width / small.width,
                            im.height / small.height,
                        ]
                    )
                    rec = {
                        "image": key,
                        "boxes": [[round(v, 2) for v in b] for b in (boxes * factor).tolist()],
                        "scores": [round(v, 5) for v in scores.tolist()],
                    }
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
            print(f"  finished after {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
