"""Score an exported D-FINE model on a set of 640 px tiles (Phase 2 plan step G0).

Usage: python gpu/eval_tiles.py --onnx last.onnx --ann holdout.json --images DIR --out FILE

Runs the model on every tile in a COCO label file (GPU if ONNX Runtime has CUDA, else CPU) and
reports, with the project's own scoring code (herdsight.evaluate):
  * centre AP over similarity thresholds 0.50:0.95 (kappa from configs/evaluation.yaml);
  * centre-matched recall and precision, and count MAE per tile, at the score threshold that
    minimises count MAE on this set (the same rule for every model compared, so the comparison
    is fair; it is not an out-of-fold operating point);
  * upright-box AP50.
Tile-level scores are for comparing training settings on a training-side hold-out only; reported
S1/S2 numbers always come from full images (scripts/score_baseline.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from herdsight.detect.dfine_onnx import CATTLE_LABEL, preprocess  # noqa: E402
from herdsight.evaluate.centre_match import (  # noqa: E402
    ImageDetections,
    average_precision,
    box_centres,
    match_hungarian,
    short_side,
)
from herdsight.evaluate.upright_ap import ImageBoxes, upright_ap  # noqa: E402
from herdsight.utils.config import load_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--ann", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    import onnxruntime as ort

    providers = [
        p
        for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if p in ort.get_available_providers()
    ]
    sess = ort.InferenceSession(str(args.onnx), providers=providers)
    coco = json.loads(args.ann.read_text())
    truth: dict[int, list] = {im["id"]: [] for im in coco["images"]}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        truth[a["image_id"]].append([x, y, x + w, y + h])
    preds = {}
    for im in coco["images"]:
        x, sizes = preprocess(Image.open(args.images / im["file_name"]), 640)
        labels, boxes, scores = sess.run(None, {"images": x, "orig_target_sizes": sizes})
        keep = labels[0] == CATTLE_LABEL
        preds[im["id"]] = (boxes[0][keep].astype(float), scores[0][keep].astype(float))
    ids = [im["id"] for im in coco["images"]]
    true = {i: np.asarray(truth[i], float).reshape(-1, 4) for i in ids}
    ev = load_config("evaluation")
    kappa = float(ev.require("centre_kappa"))
    sim = float(ev.require("centre_match_similarity"))
    start, stop, step = ev.require("operating_threshold_grid")
    grid = np.round(np.arange(start, stop + 1e-9, step), 4)
    n_true = np.array([len(true[i]) for i in ids], float)
    maes = [
        np.mean(np.abs(np.array([(preds[i][1] >= t).sum() for i in ids]) - n_true)) for t in grid
    ]
    thr = float(grid[int(np.argmin(maes))])
    tp = n_pred = 0
    for i in ids:
        b, s = preds[i]
        m = s >= thr
        n_pred += int(m.sum())
        if len(true[i]):
            tp += match_hungarian(
                box_centres(b[m]),
                box_centres(true[i]),
                np.maximum(short_side(true[i]), 1.0),
                kappa,
                sim,
            ).tp
    imgs = [
        ImageDetections.make(
            box_centres(preds[i][0]),
            preds[i][1],
            box_centres(true[i]),
            np.maximum(short_side(true[i]), 1.0),
        )
        for i in ids
    ]
    out = {
        "onnx": str(args.onnx),
        "annotations": str(args.ann),
        "tiles": len(ids),
        "animals": int(n_true.sum()),
        "providers": sess.get_providers(),
        "threshold_min_count_mae_on_this_set": thr,
        "count_mae_per_tile": float(min(maes)),
        "centre_matched_recall": tp / max(n_true.sum(), 1),
        "centre_matched_precision": tp / max(n_pred, 1),
        "centre_ap": average_precision(imgs, kappa)["ap_mean"],
        "upright_ap50": upright_ap([ImageBoxes.make(*preds[i], true[i]) for i in ids])["ap50"],
    }
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
