"""Run a baseline detector over one cross-validation fold; save predictions to storage (D-26).

Usage:
  python scripts/run_baseline.py --detector zeroshot --fold all
  python scripts/run_baseline.py --detector dfine --fold fold0 --onnx <fold0 last.onnx>
  python scripts/run_baseline.py --detector dfine --onboard --fold fold0 --onnx <onboard fold0>
  python scripts/run_baseline.py --detector dfine --fold fold0 --onnx <model> --name dfine_g2
  python scripts/run_baseline.py --detector dfine --fold all --farm Derval --onnx <model> \
      --name dfine_g6ref_Derval

--farm with --fold all predicts every cross-validation image of one farm, for a leave-one-farm-out
model that was trained without that farm (D-33).

Predictions go to $HERDSIGHT_STORAGE/runs/baselines/<detector>/<fold>.jsonl, one line per image
(boxes in full-image pixels, scores). A run that stops part-way resumes where it left off.
D-FINE's model for fold k was trained without fold k, so its predictions are out-of-fold. The
zero-shot detector has no cattle training, so it runs on every fold, on the fixed image sample in
configs/baselines.yaml (D-24). Test-farm images are never read.

--onboard runs the S7 herd detector (D-29): each still is first reduced to onboard_long_side_px,
exactly as its training tiles were, tiled with onboard_tile_overlap_px, and the boxes are scaled
back to full-image pixels. Its predictions go to runs/baselines/dfine_onboard/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime

from PIL import Image

from herdsight.data.icaerus import default_root, iter_images
from herdsight.data.splits import sample_images
from herdsight.detect.run_tiled import detect_full_image
from herdsight.utils.config import REPO_ROOT, load_config
from herdsight.utils.env_report import env_report
from herdsight.utils.io import load_rgb
from herdsight.utils.paths import storage_path


def image_key(s: object) -> str:
    return f"{s.farm}/{s.flight}/{s.path.name}"  # type: ignore[attr-defined]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", choices=["zeroshot", "dfine"], required=True)
    ap.add_argument("--fold", required=True, help="fold0, fold1, ... or 'all' (zero-shot only)")
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--onboard", action="store_true", help="S7 herd detector on reduced stills")
    ap.add_argument(
        "--name", default=None, help="runs/baselines subfolder (default: the detector's name)"
    )
    ap.add_argument("--gpu", action="store_true", help="D-FINE on CUDA; stops if it is unavailable")
    ap.add_argument("--farm", default=None, help="only this farm (leave-one-farm-out models)")
    ap.add_argument(
        "--long-side", type=int, default=None, help="--onboard: long side to reduce stills to"
    )
    args = ap.parse_args(argv)
    if args.farm and (args.detector != "dfine" or args.fold != "all" or args.onboard):
        ap.error("--farm applies to D-FINE tile models with --fold all")
    cfg = load_config("baselines")
    overlap = int(cfg.require("tile_overlap_px"))
    long_side = None
    if args.onboard:
        if args.detector != "dfine":
            ap.error("--onboard applies to the dfine detector only")
        overlap = int(cfg.require("onboard_tile_overlap_px"))
        long_side = args.long_side or int(cfg.require("onboard_long_side_px"))
    run_name = args.name or ("dfine_onboard" if args.onboard else args.detector)
    sp = json.loads((REPO_ROOT / "results" / "splits.json").read_text())["icaerus_v2"]
    folds = [f"fold{k}" for k in range(sp["n_folds"])]
    wanted = folds if args.fold == "all" else [args.fold]
    all_allowed = args.detector == "zeroshot" or args.farm
    if any(f not in folds for f in wanted) or (args.fold == "all" and not all_allowed):
        ap.error(f"--fold must be one of {folds}, or 'all' for the zero-shot detector or --farm")
    fold_of: dict[str, str] = {}
    images = []
    for s, problem in iter_images(default_root(), with_metadata=False):
        fold = sp["flights"].get(f"{s.farm}/{s.flight}")
        if fold in wanted and problem is None and (not args.farm or s.farm == args.farm):
            images.append(s)
            fold_of[image_key(s)] = fold
    if args.detector == "zeroshot":
        from herdsight.detect.owlv2 import Owlv2TileDetector

        keep = set(
            sample_images(
                [image_key(s) for s in images],
                float(cfg.require("zeroshot_sample_share")),
                int(cfg.require("sample_seed")),
            )
        )
        images = [s for s in images if image_key(s) in keep]
        tile = int(cfg.require("zeroshot_tile_px"))
        detector = Owlv2TileDetector(
            cfg.require("zeroshot_model"),
            cfg.require("zeroshot_revision"),
            cfg.require("zeroshot_queries"),
            float(cfg.require("zeroshot_min_score")),
            threads=args.threads or None,
        )
    else:
        from herdsight.detect.dfine_onnx import DfineOnnxTileDetector

        if not args.onnx:
            ap.error("--onnx is required for the dfine detector")
        tile = int(cfg.require("dfine_tile_px"))
        detector = DfineOnnxTileDetector(
            args.onnx, min_score=0.01, size=tile, threads=args.threads, gpu=args.gpu
        )

    run_dir = storage_path("runs", "baselines", run_name, create=True)
    done = set()
    for fold in wanted:
        path = run_dir / f"{fold}.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    done.add(json.loads(line)["image"])
                except (json.JSONDecodeError, KeyError):
                    pass  # a line cut short by an interruption
    todo = [s for s in sorted(images, key=image_key) if image_key(s) not in done]
    print(f"{args.detector}/{args.fold}: {len(images)} images, {len(todo)} to do", flush=True)
    t_all = time.time()
    handles = {fold: (run_dir / f"{fold}.jsonl").open("a", encoding="utf-8") for fold in wanted}
    try:
        for i, s in enumerate(todo, 1):
            f = handles[fold_of[image_key(s)]]
            t = time.time()
            im = load_rgb(s.path)
            scale = 1.0
            if long_side:
                scale = long_side / max(im.size)
                im = im.resize(
                    (round(im.width * scale), round(im.height * scale)), Image.Resampling.LANCZOS
                )
            boxes, scores = detect_full_image(im, detector, tile, overlap)
            boxes = boxes / scale
            rec = {
                "image": image_key(s),
                "boxes": [[round(v, 2) for v in b] for b in boxes.tolist()],
                "scores": [round(v, 5) for v in scores.tolist()],
                "seconds": round(time.time() - t, 2),
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 5 == 0 or i == len(todo):
                rate = (time.time() - t_all) / i
                print(f"  {i}/{len(todo)}  {rate:.0f} s/image", flush=True)
    finally:
        for h in handles.values():
            h.close()
    record = {
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "detector": run_name,
        "fold": args.fold,
        "farm": args.farm,
        "onboard_long_side_px": long_side,
        "n_images": len(images),
        "tile_px": tile,
        "overlap_px": overlap,
        "onnx": args.onnx,
        "predictions": [str(run_dir / f"{fold}.jsonl") for fold in wanted],
        "environment": env_report(),
    }
    dest = REPO_ROOT / "results" / "baselines" / f"{run_name}_{args.fold}_run.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
