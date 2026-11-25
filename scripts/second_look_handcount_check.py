"""Second look: the full chain (OWLv2 detections -> box tracker -> watch count) against hand counts.

Usage: python scripts/second_look_handcount_check.py   (after scripts/sheep_watch_detect.py and
       the owner's hand_counts.csv)

Rule fixed before any count was read (_private notes, 24 Nov): the OWLv2 score threshold is the
value from 0.05 to 0.50 (step 0.05) with the best F1 on the four annotated clips (a detection
matches an unmatched true box that contains its centre); on the four hand-counted clips, boxes at
or above it are tracked with herdsight.count.watch.track_boxes defaults (dt 0.2 s at 5 fps) after
converting pixels to metres with the median detected box length taken as 1.2 m, and counted with
watch_count defaults. Truth is the mean of the owner's two counts (or the third, if noted). Pass:
the watched count's mean absolute error is smaller than the best single frame's (most detections
in one frame), and the hand count is inside the watched range for at least 3 of the 4 clips.
Sheep, not cattle; tests the watching method with a real detector. Writes
results/phase5/watch_count_handcount_check.json.
"""

from __future__ import annotations

import csv
import glob
import json
from collections import defaultdict

import numpy as np

from herdsight.count.watch import track_boxes, watch_count
from herdsight.utils.config import REPO_ROOT
from herdsight.utils.paths import storage_path

ANNOTATED = ("crop_23.11.23-8", "crop_23.11.23-10", "crop_23.11.23-12", "crop_23.11.23-13")
HAND = (
    "crop_23.11.23-2_15fps",
    "crop_23.11.23-4_15fps",
    "crop_23.11.23-9_15fps",
    "crop_23.11.23-11_15fps",
)
THRESHOLDS = np.round(np.arange(0.05, 0.501, 0.05), 2)
ANIMAL_M = 1.2


def detections(name: str) -> dict:
    return json.loads((storage_path("runs", "second_look") / f"owlv2_{name}.json").read_text())


def truth_boxes(clip: str) -> dict[int, np.ndarray]:
    root = storage_path("raw", "icaerus_sheep_annotated", "unpacked", "depot_zenodo_2_annotations")
    f = glob.glob(str(root / "MOT1.1" / f"*{clip}*" / "**" / "gt.txt"), recursive=True)[0]
    out: dict[int, list] = defaultdict(list)
    for line in open(f):
        if line.strip():
            fr, _, x, y, w, h = (float(v) for v in line.split(",")[:6])
            out[int(fr)].append([x, y, x + w, y + h])
    return {k: np.array(v) for k, v in out.items()}


def f1_at(t: float) -> float:
    tp = fp = fn = 0
    for clip in ANNOTATED:
        gt = truth_boxes(clip)
        for fr in detections(clip)["detections"]:
            b = np.array(fr["boxes"]).reshape(-1, 4)[np.array(fr["scores"]) >= t]
            g = gt.get(fr["frame"], np.zeros((0, 4)))
            used = set()
            for box in b:
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                hit = [
                    i
                    for i in range(len(g))
                    if i not in used and g[i, 0] <= cx <= g[i, 2] and g[i, 1] <= cy <= g[i, 3]
                ]
                if hit:
                    used.add(hit[0])
                    tp += 1
                else:
                    fp += 1
            fn += len(g) - len(used)
    return 2 * tp / max(2 * tp + fp + fn, 1)


def hand_counts() -> dict[str, float]:
    path = (
        storage_path("raw", "icaerus_sheep_counting_videos", "for_hand_count") / "hand_counts.csv"
    )
    out = {}
    for row in csv.DictReader(open(path, encoding="utf-8-sig")):
        clip = row["clip"].removesuffix(".MP4")
        counts = [float(row[k]) for k in ("count_1", "count_2") if row.get(k, "").strip()]
        note = row.get("notes", "").strip()
        third = [float(w) for w in note.replace(",", " ").split() if w.replace(".", "").isdigit()]
        if len(counts) == 2 and abs(counts[0] - counts[1]) > 1 and third:
            out[clip] = third[0]
        elif counts:
            out[clip] = float(np.mean(counts))
    return out


def main() -> int:
    f1 = {float(t): f1_at(float(t)) for t in THRESHOLDS}
    thr = max(f1, key=f1.get)
    truth = hand_counts()
    missing = [c for c in HAND if c not in truth]
    if missing:
        raise SystemExit(f"hand counts missing for {missing}")
    rows = []
    for clip in HAND:
        d = detections(clip)
        dt = d["every"] / d["fps"]
        frames = []
        for fr in d["detections"]:
            b = np.array(fr["boxes"]).reshape(-1, 4)
            frames.append(b[np.array(fr["scores"]) >= thr])
        all_b = np.vstack([b for b in frames if len(b)])
        px_per_m = (
            float(np.median(np.maximum(all_b[:, 2] - all_b[:, 0], all_b[:, 3] - all_b[:, 1])))
            / ANIMAL_M
        )
        tracks, dropped = track_boxes([b / px_per_m for b in frames], dt_s=dt)
        w = watch_count(tracks, len(frames))
        rows.append(
            {
                "clip": clip,
                "hand_count": truth[clip],
                "watched": w["count"],
                "low": w["low"],
                "high": w["high"],
                "single_frame_best": max(len(b) for b in frames),
                "in_range": w["low"] <= truth[clip] <= w["high"],
                "px_per_m": px_per_m,
                "tentative_dropped": dropped,
                "frames": len(frames),
                "dt_s": dt,
            }
        )
    mae_w = float(np.mean([abs(r["watched"] - r["hand_count"]) for r in rows]))
    mae_s = float(np.mean([abs(r["single_frame_best"] - r["hand_count"]) for r in rows]))
    in_range = sum(r["in_range"] for r in rows)
    out = {
        "note": "Public sheep video (Zenodo 18889878, CC BY 4.0), hand counts; OWLv2 zero-shot "
        "detections; sheep, not cattle. Not a field result.",
        "threshold_f1_on_annotated": f1,
        "threshold": thr,
        "watched_mae": mae_w,
        "single_frame_mae": mae_s,
        "clips_in_range": in_range,
        "passes": mae_w < mae_s and in_range >= 3,
        "rows": rows,
    }
    dest = REPO_ROOT / "results" / "phase5" / "watch_count_handcount_check.json"
    dest.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "threshold_f1_on_annotated"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
