"""Make the blind re-annotation task that measures label noise and sets kappa (brief 9.1).

Usage: python scripts/make_kappa_task.py

Picks `N_TASKS` cattle at random (seeded) from the cross-validation folds (never the test
farm), cuts a square picture
around each, shifted at random so the original box position cannot be inferred from the crop,
and writes to $HERDSIGHT_STORAGE/exports/kappa_task/:
  index.html        the annotation page (open in a browser; no internet needed)
  crops/*.jpg       the pictures
  key.json          where each crop came from (not shown to the annotator)
The annotator draws a box on the animal nearest the middle of each picture and downloads
centre_answers.json; scripts/fit_kappa.py turns that into kappa.
"""

from __future__ import annotations

import json
import sys

import numpy as np
from PIL import Image

from herdsight.data.icaerus import default_root, iter_images
from herdsight.utils.config import REPO_ROOT
from herdsight.utils.paths import storage_path

N_TASKS = 100
SEED = 20261118
CROP_FACTOR = 3.0  # crop side = 3 x the animal's long side
MAX_SHIFT = 0.2  # random shift of the animal from the crop centre, as a share of the side
DISPLAY = 800


def main() -> int:
    splits = json.loads((REPO_ROOT / "results" / "splits.json").read_text())["icaerus_v2"][
        "flights"
    ]
    root = default_root()
    cands = []
    for s, problem in iter_images(root, with_metadata=False):
        if problem or not s.labelled or splits.get(f"{s.farm}/{s.flight}", "test") == "test":
            continue
        for j in range(s.boxes_xyxy.shape[0]):
            cands.append((s.path, s.boxes_xyxy, j))
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(cands), size=min(N_TASKS, len(cands)), replace=False)
    out = storage_path("exports", "kappa_task", create=True)
    (out / "crops").mkdir(exist_ok=True)
    tasks, key = [], {}
    for n, idx in enumerate(sorted(pick.tolist())):
        path, boxes, j = cands[idx]
        x0, y0, x1, y1 = boxes[j]
        side = CROP_FACTOR * max(x1 - x0, y1 - y0)
        cx = (x0 + x1) / 2 + rng.uniform(-MAX_SHIFT, MAX_SHIFT) * side
        cy = (y0 + y1) / 2 + rng.uniform(-MAX_SHIFT, MAX_SHIFT) * side
        ox, oy = cx - side / 2, cy - side / 2
        with Image.open(path) as im:
            crop = im.convert("RGB").crop(
                (round(ox), round(oy), round(ox + side), round(oy + side))
            )
        crop = crop.resize((DISPLAY, DISPLAY), Image.Resampling.LANCZOS)
        tid = f"t{n:03d}"
        crop.save(out / "crops" / f"{tid}.jpg", quality=92)
        tasks.append({"id": tid, "file": f"crops/{tid}.jpg", "size": DISPLAY})
        scale = DISPLAY / side
        key[tid] = {
            "image": str(path),
            "origin_px": [round(ox), round(oy)],
            "scale": scale,
            # every labelled box inside the crop, in display pixels (nearest one is matched)
            "boxes_display": [
                [
                    (b[0] - round(ox)) * scale,
                    (b[1] - round(oy)) * scale,
                    (b[2] - round(ox)) * scale,
                    (b[3] - round(oy)) * scale,
                ]
                for b in boxes.tolist()
                if round(ox) <= (b[0] + b[2]) / 2 <= round(ox) + side
                and round(oy) <= (b[1] + b[3]) / 2 <= round(oy) + side
            ],
            "target_index_in_image": j,
        }
    html = (REPO_ROOT / "scripts" / "assets" / "centre_annotator.html").read_text(encoding="utf-8")
    (out / "index.html").write_text(html.replace("__TASKS__", json.dumps(tasks)), encoding="utf-8")
    (out / "key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    print(f"{len(tasks)} tasks written to {out}; open {out / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
