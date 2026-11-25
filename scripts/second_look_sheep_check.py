"""Second look: the watch counter on real tracked sheep video, against the true count.

Usage: python scripts/second_look_sheep_check.py [--measured-scale]

Data: ICAERUS "Drone videos and their annotations of passing sheep" (Zenodo 12094356, CC BY 4.0;
Helary, Lebreton et al.), four clips with tracked boxes (MOT1.1 gt.txt), 9 fps, filmed from 5 to
10 m. The true count of a clip is its number of distinct identities. The clips' true boxes are
degraded as a detector fails (missed animals, more often when touching; touching pairs merged
into one; occasional false detections; jitter), with fixed seeds, and passed through
herdsight.count.watch. The rule was fixed before any run: over all clip-seed pairs, the watched
count's mean absolute error must be smaller than that of the best single photo (the frame with
the most detections), and the true count must be inside the watched range in at least 80% of
pairs. These are sheep walking through a corridor, not grazing cattle: the check tests the
tracking and counting rules, not the cattle detector. Writes
results/phase5/watch_count_sheep_check.json.

--measured-scale is a diagnosis added after that run failed: the clips were cropped and resized,
so the pixels-per-metre guessed from the flight heights were wrong (by 1.1 to 4 times; the 5 m
clip's 1.5 m merge distance spanned about five sheep). It takes each clip's scale from its own
animals (median box length taken as 1.2 m, a typical adult sheep) and writes
watch_count_sheep_check_measured_scale.json. Same rule, same seeds, no other change; it does not
replace the failed result, which stays on record. On a drone the scale is exact (flight height
and camera), so this is a fault of the test set-up, not of the method.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict

import numpy as np

from herdsight.count.watch import track_positions, watch_count
from herdsight.utils.config import REPO_ROOT
from herdsight.utils.paths import storage_path

CLIPS = {  # clip: pixels per metre (about 160 at 10 m, 320 at 5 m; summary.xlsx heights)
    "crop_23.11.23-8": 160.0,
    "crop_23.11.23-10": 160.0,
    "crop_23.11.23-12": 160.0,
    "crop_23.11.23-13": 320.0,
}
DT, GATE_M, SEEDS = 1 / 9, 0.5, range(10)
P_MISS, P_MISS_TOUCH, P_MERGE, P_FALSE, JITTER_PX = 0.15, 0.30, 0.2, 0.05, 4.0
W, H = 1440, 1080


def load_gt(clip: str) -> tuple[dict[int, np.ndarray], int]:
    root = storage_path("raw", "icaerus_sheep_annotated", "unpacked", "depot_zenodo_2_annotations")
    files = glob.glob(str(root / "MOT1.1" / f"*{clip}*" / "**" / "gt.txt"), recursive=True)
    if len(files) != 1:
        raise SystemExit(f"{clip}: expected one gt.txt, found {len(files)}")
    boxes: dict[int, list[list[float]]] = defaultdict(list)
    ids = set()
    for line in open(files[0]):
        if not line.strip():
            continue
        f, i, x, y, w, h = line.split(",")[:6]
        boxes[int(f)].append([float(x), float(y), float(x) + float(w), float(y) + float(h)])
        ids.add(i)
    n = max(boxes)
    return {f: np.array(boxes.get(f, []), float).reshape(-1, 4) for f in range(1, n + 1)}, len(ids)


def touching(b: np.ndarray, pad: float = 2.0) -> np.ndarray:
    """Pairs (i, j) of boxes that overlap or nearly touch."""
    pairs = []
    for i in range(len(b)):
        for j in range(i + 1, len(b)):
            if not (
                b[i, 2] + pad < b[j, 0]
                or b[j, 2] + pad < b[i, 0]
                or b[i, 3] + pad < b[j, 1]
                or b[j, 3] + pad < b[i, 1]
            ):
                pairs.append((i, j))
    return np.array(pairs, int).reshape(-1, 2)


def degrade(boxes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Detected centres (pixels) from one frame's true boxes."""
    c = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2]
    if not len(c):
        return np.zeros((0, 2))
    pairs = touching(boxes)
    touch = np.zeros(len(c), bool)
    touch[pairs.ravel()] = True
    keep = rng.random(len(c)) >= np.where(touch, P_MISS_TOUCH, P_MISS)
    out, merged = [], set()
    for i, j in pairs:
        if keep[i] and keep[j] and i not in merged and j not in merged and rng.random() < P_MERGE:
            out.append((c[i] + c[j]) / 2)
            merged.update((i, j))
    out += [c[k] for k in range(len(c)) if keep[k] and k not in merged]
    if rng.random() < P_FALSE:
        out.append(rng.uniform([0, 0], [W, H]))
    pts = np.array(out, float).reshape(-1, 2)
    return pts + rng.normal(0, JITTER_PX, pts.shape)


def measured_px_per_m(gt: dict[int, np.ndarray], animal_m: float = 1.2) -> float:
    b = np.vstack([v for v in gt.values() if len(v)])
    return float(np.median(np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]))) / animal_m


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measured-scale", action="store_true")
    args = ap.parse_args(argv)
    rows = []
    scales = {}
    for clip, px_per_m in CLIPS.items():
        gt, truth = load_gt(clip)
        if args.measured_scale:
            px_per_m = measured_px_per_m(gt)
        scales[clip] = px_per_m
        frames = [gt[f] for f in sorted(gt)]
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            det = [degrade(b, rng) for b in frames]
            tracks = track_positions([d / px_per_m for d in det], dt_s=DT, gate_m=GATE_M)
            w = watch_count(tracks, len(det))
            single = max(len(d) for d in det)
            rows.append(
                {
                    "clip": clip,
                    "seed": seed,
                    "true_count": truth,
                    "watched": w["count"],
                    "low": w["low"],
                    "high": w["high"],
                    "single_photo_best": single,
                    "true_in_range": w["low"] <= truth <= w["high"],
                    "switches_merged": w["switches_merged"],
                }
            )
    err_w = float(np.mean([abs(r["watched"] - r["true_count"]) for r in rows]))
    err_s = float(np.mean([abs(r["single_photo_best"] - r["true_count"]) for r in rows]))
    cover = float(np.mean([r["true_in_range"] for r in rows]))
    by_clip = {}
    for clip in CLIPS:
        rs = [r for r in rows if r["clip"] == clip]
        by_clip[clip] = {
            "true_count": rs[0]["true_count"],
            "watched_mean": float(np.mean([r["watched"] for r in rs])),
            "watched_mae": float(np.mean([abs(r["watched"] - r["true_count"]) for r in rs])),
            "single_photo_mae": float(
                np.mean([abs(r["single_photo_best"] - r["true_count"]) for r in rs])
            ),
            "true_in_range_share": float(np.mean([r["true_in_range"] for r in rs])),
        }
    out = {
        "note": (
            "Public data (ICAERUS passing-sheep video, Zenodo 12094356, CC BY 4.0), true boxes "
            "degraded as a detector fails; sheep walking through a corridor, not grazing cattle, "
            "and not the cattle detector. Not a field result."
        ),
        "degradation": {
            "miss": P_MISS,
            "miss_touching": P_MISS_TOUCH,
            "merge_touching_pair": P_MERGE,
            "false_per_frame": P_FALSE,
            "jitter_px": JITTER_PX,
        },
        "tracker": {"dt_s": DT, "gate_m": GATE_M},
        "pixels_per_metre": scales,
        "scale": "measured from the animals (diagnosis after the failed run)"
        if args.measured_scale
        else "guessed from flight height (the rule as fixed)",
        "pairs": len(rows),
        "watched_mae": err_w,
        "single_photo_mae": err_s,
        "true_in_watched_range_share": cover,
        "passes": err_w < err_s and cover >= 0.8,
        "by_clip": by_clip,
        "rows": rows,
    }
    name = "watch_count_sheep_check" + ("_measured_scale" if args.measured_scale else "")
    dest = REPO_ROOT / "results" / "phase5" / f"{name}.json"
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "watched_mae",
                    "single_photo_mae",
                    "true_in_watched_range_share",
                    "passes",
                    "by_clip",
                )
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
