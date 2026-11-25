"""Second look: OWLv2 sheep detections on the annotated and the hand-counted sheep clips.

Usage (an environment with torch and transformers; HF_HOME on the storage drive):
  python scripts/sheep_watch_detect.py

Runs the project's zero-shot detector (herdsight.detect.owlv2, OWLv2 base ensemble, revision
pinned in configs/baselines.yaml, Apache-2.0) with the queries "a photo of a sheep" and "a sheep"
on each frame, resized so the long side is 960 px (the model's input size), keeping scores of at
least 0.05. Annotated clips (Zenodo 12094356, 9 fps): every 2nd frame, used only to choose the
score threshold. Hand-counted clips (Zenodo 18889878, 15 fps): every 3rd frame (5 fps), the test.
Boxes are saved in each clip's own pixels, with the frame number, to
$HERDSIGHT_STORAGE/runs/second_look/owlv2_<clip>.json. Rule fixed beforehand
(_private notes, 24 Nov): no setting here is changed after any result is seen.
"""

from __future__ import annotations

import json
import time

import cv2
from PIL import Image

from herdsight.detect.owlv2 import Owlv2TileDetector
from herdsight.utils.config import load_config
from herdsight.utils.paths import storage_path

QUERIES = ("a photo of a sheep", "a sheep")
MIN_SCORE, LONG_SIDE = 0.05, 960
ANNOTATED = ("crop_23.11.23-8", "crop_23.11.23-10", "crop_23.11.23-12", "crop_23.11.23-13")
HAND = (
    "crop_23.11.23-2_15fps",
    "crop_23.11.23-4_15fps",
    "crop_23.11.23-9_15fps",
    "crop_23.11.23-11_15fps",
)


def run(det: Owlv2TileDetector, video: str, every: int, dest: str) -> None:
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames, f, t0 = [], 0, time.time()
    while True:
        ok, im = cap.read()
        if not ok:
            break
        f += 1  # 1-based, as MOT frame numbers
        if (f - 1) % every:
            continue
        h, w = im.shape[:2]
        s = LONG_SIDE / max(w, h)
        small = Image.fromarray(
            cv2.cvtColor(cv2.resize(im, (round(w * s), round(h * s))), cv2.COLOR_BGR2RGB)
        )
        boxes, scores = det(small)
        frames.append(
            {"frame": f, "boxes": (boxes / s).round(1).tolist(), "scores": scores.round(4).tolist()}
        )
    out = {
        "video": video.replace("\\", "/").rsplit("/", 1)[-1],
        "fps": fps,
        "every": every,
        "frames_total": f,
        "queries": list(QUERIES),
        "min_score": MIN_SCORE,
        "long_side": LONG_SIDE,
        "seconds": round(time.time() - t0, 1),
        "detections": frames,
    }
    with open(dest, "w") as fh:
        json.dump(out, fh)
    print(f"{out['video']}: {len(frames)} frames in {out['seconds']} s")


def main() -> int:
    cfg = load_config("baselines")
    det = Owlv2TileDetector(
        cfg.require("zeroshot_model"), cfg.require("zeroshot_revision"), QUERIES, MIN_SCORE
    )
    out = storage_path("runs", "second_look", create=True)
    ann = storage_path(
        "raw", "icaerus_sheep_annotated", "unpacked", "depot_zenodo_2_annotations", "Videos"
    )
    hand = storage_path("raw", "icaerus_sheep_counting_videos", "for_hand_count")
    jobs = [(str(ann / f"{c}.MP4"), 2, c) for c in ANNOTATED]
    jobs += [(str(hand / f"{c}.MP4"), 3, c) for c in HAND]
    for video, every, name in jobs:
        dest = out / f"owlv2_{name}.json"
        if dest.exists():
            continue
        run(det, video, every, str(dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
