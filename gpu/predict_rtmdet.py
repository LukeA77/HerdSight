"""Phase 2 step G7: out-of-fold, full-image predictions of the RTMDet-R models.

Usage: python gpu/predict_rtmdet.py --models DIR [--name rtmdet_r] [--device cuda] [--limit N]

<DIR>/g7_fold<k>/ holds the model trained without fold k. Images are tiled exactly as for D-FINE and
detections are kept by the tile that owns their centre. Output, under
$HERDSIGHT_STORAGE/runs/baselines/<name>/<fold>.jsonl: "boxes" (the upright box around each oriented
box, so scripts/phase2_compare.py scores it like any other model), "scores" and "obbs". A run that
stops part-way resumes. Test-farm images are never read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from mmdet.apis import init_detector
from mmdet.structures import DetDataSample
from mmrotate.utils import register_all_modules

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import py39_compat  # noqa: E402, F401  (before any herdsight module)
from herdsight.data.icaerus import default_root, iter_images  # noqa: E402
from herdsight.detect.tiling import tile_grid  # noqa: E402
from herdsight.labels.obb_tiles import enclosing_upright  # noqa: E402
from herdsight.utils.config import REPO_ROOT, load_config  # noqa: E402
from herdsight.utils.io import load_rgb  # noqa: E402
from herdsight.utils.paths import storage_path  # noqa: E402

BATCH = 16


def predict_image(model, rgb: np.ndarray, tile: int, overlap: int, device: str):  # noqa: ANN001, ANN201
    h, w = rgb.shape[:2]
    bgr = rgb[:, :, ::-1]
    tiles = list(tile_grid(w, h, tile, overlap))
    all_obb, all_scores = [], []
    for start in range(0, len(tiles), BATCH):
        part = tiles[start : start + BATCH]
        inputs, samples = [], []
        for t in part:
            crop = bgr[t.y0 : t.y1, t.x0 : t.x1].copy()
            inputs.append(torch.from_numpy(crop).permute(2, 0, 1))
            shape = crop.shape[:2]
            samples.append(
                DetDataSample(
                    metainfo={
                        "img_shape": shape,
                        "ori_shape": shape,
                        "pad_shape": shape,
                        "scale_factor": (1.0, 1.0),
                    }
                )
            )
        data = model.data_preprocessor({"inputs": inputs, "data_samples": samples}, False)
        with torch.no_grad():
            results = model(**data, mode="predict")
        for t, r in zip(part, results, strict=True):
            obb = r.pred_instances.bboxes
            obb = (obb.tensor if hasattr(obb, "tensor") else obb).cpu().numpy().astype(float)
            scores = r.pred_instances.scores.cpu().numpy().astype(float)
            obb = obb.reshape(-1, 5).copy()
            obb[:, 0] += t.x0
            obb[:, 1] += t.y0
            keep = t.owns(obb[:, 0], obb[:, 1])
            all_obb.append(obb[keep])
            all_scores.append(scores[keep])
    return np.concatenate(all_obb), np.concatenate(all_scores)


def done_images(path: Path) -> set:
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["image"])
            except (json.JSONDecodeError, KeyError):
                pass  # a line cut short by an interruption
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--name", default="rtmdet_r")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    register_all_modules()
    base = load_config("baselines")
    tile, overlap = int(base.require("dfine_tile_px")), int(base.require("tile_overlap_px"))
    sp = json.loads((REPO_ROOT / "results" / "splits.json").read_text())["icaerus_v2"]
    folds = [f"fold{k}" for k in range(sp["n_folds"])]  # the split file also lists test flights
    by_fold: dict = {f: [] for f in folds}
    for s, problem in iter_images(default_root(), with_metadata=False):
        fold = sp["flights"].get(f"{s.farm}/{s.flight}")
        if fold in by_fold and problem is None and s.labelled:
            by_fold[fold].append((f"{s.farm}/{s.flight}/{s.path.name}", s.path))
    out = storage_path("runs", "baselines", args.name, create=True)
    t0 = time.time()
    for k, fold in enumerate(folds):
        d = args.models / f"g7_fold{k}"
        # a palette is passed because the saved config has no test dataset to read one from
        model = init_detector(
            str(d / "config.py"), str(d / "last.pth"), palette="random", device=args.device
        )
        model.eval()
        rows = sorted(by_fold[fold], key=lambda r: r[0])
        if args.limit:
            rows = rows[: args.limit]
        path = out / f"{fold}.jsonl"
        done = done_images(path)
        todo = [r for r in rows if r[0] not in done]
        print(f"{fold}: {len(todo)} of {len(rows)} to do", flush=True)
        with path.open("a", encoding="utf-8") as fh:
            for i, (key, image_path) in enumerate(todo, 1):
                rgb = np.asarray(load_rgb(image_path))
                obb, scores = predict_image(model, rgb, tile, overlap, args.device)
                rec = {
                    "image": key,
                    "boxes": [[round(v, 2) for v in b] for b in enclosing_upright(obb).tolist()],
                    "scores": [round(v, 5) for v in scores.tolist()],
                    "obbs": [[round(v, 4) for v in o] for o in obb.tolist()],
                }
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                if i % 20 == 0 or i == len(todo):
                    print(f"  {fold} {i}/{len(todo)}  {time.time() - t0:.0f} s", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
