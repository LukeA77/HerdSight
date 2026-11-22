"""Make the owner's review task that turns candidate animal links into S3 ground truth (brief 8.3).

Usage: python scripts/make_link_review.py [--round 2|3]

Flights reviewed (each rule fixed before any detector result was scored on S3):
* round 1: in each cross-validation fold, the flight with the most labelled animals among
  flights with at least 5 registered image pairs;
* round 2 (added 2026-11-19, after round 1 showed the unverified candidates are 6-21% off, so
  only verified flights can be scored): in each farm-date group not covered by round 1, the
  flight with the most labelled animals among flights with at least 5 registered image pairs;
* round 3 (added 2026-11-22, Phase 3 plan): every remaining flight, so that all 20 flights have
  verified counts and S3 can be fitted and scored out of fold with at least five flights per
  fold. Flights without overlapping images get no questions; their count is the labels'.
Round 1 is written to exports/link_review/, later rounds to exports/link_review_<round>/.

Two kinds of question, written to $HERDSIGHT_STORAGE/exports/link_review/ (index.html + pictures):

* **Same animal?** For each candidate animal seen in two or more images, a row of pictures of
  it, one per image. The reviewer unticks any picture that shows a different animal from the
  first.
* **Missed link?** For each animal seen in only one image, but inside the area of another,
  registered image: its picture beside the matching place in the other image (crosshair), with
  that image's labelled boxes drawn. The reviewer answers: it is one of the boxes there / it is
  there but has no box / it is not there (moved, hidden or out of view).

scripts/build_s3_truth.py applies the answers.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from PIL import Image, ImageDraw

from herdsight.data.icaerus import default_root
from herdsight.data.splits import FlightStats
from herdsight.geo.register import apply_homography
from herdsight.utils.config import REPO_ROOT
from herdsight.utils.io import load_rgb
from herdsight.utils.paths import storage_path

THUMB = 220
MIN_REGISTERED_PAIRS = 5


def crop_around(
    img: Image.Image, cx: float, cy: float, side: float
) -> tuple[Image.Image, float, float]:
    x0, y0 = cx - side / 2, cy - side / 2
    c = img.crop((round(x0), round(y0), round(x0 + side), round(y0 + side))).resize((THUMB, THUMB))
    return c, round(x0), round(y0)


def draw_boxes(
    c: Image.Image, boxes: np.ndarray, x0: float, y0: float, side: float, colour: str
) -> None:
    s = THUMB / side
    d = ImageDraw.Draw(c)
    for b in boxes:
        d.rectangle(
            [(b[0] - x0) * s, (b[1] - y0) * s, (b[2] - x0) * s, (b[3] - y0) * s],
            outline=colour,
            width=2,
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1, choices=[1, 2, 3])
    args = ap.parse_args(argv)
    summary = json.loads((REPO_ROOT / "results" / "data" / "s3_candidate_links.json").read_text())
    splits = json.loads((REPO_ROOT / "results" / "splits.json").read_text())["icaerus_v2"][
        "flights"
    ]

    def group(key: str) -> str:
        return FlightStats(*key.split("/"), 0, 0).group

    if args.round == 1:
        bucket = splits.get
        exclude: set[str] = set()
    elif args.round == 3:
        bucket = str  # every flight is its own bucket

        def reviewed(folder: str) -> set[str]:
            return set(
                json.loads((storage_path("exports", folder) / "tasks.json").read_text())[
                    "flights"
                ].values()
            )

        exclude = reviewed("link_review") | reviewed("link_review_2")
    else:
        bucket = group
        done = json.loads((storage_path("exports", "link_review") / "tasks.json").read_text())
        exclude = {group(k) for k in done["flights"].values()}
    chosen = {}
    for key, v in sorted(summary.items()):
        b = bucket(key)
        if b in exclude or (args.round < 3 and v["registered_pairs"] < MIN_REGISTERED_PAIRS):
            continue
        if b not in chosen or v["naive_sum"] > summary[chosen[b]]["naive_sum"]:
            chosen[b] = key
    links_dir = storage_path("processed", "s3_links")
    folder = "link_review" if args.round == 1 else f"link_review_{args.round}"
    out = storage_path("exports", folder, create=True)
    (out / "pics").mkdir(exist_ok=True)
    root = default_root()
    tasks = []
    for _fold, key in sorted(chosen.items()):
        farm, flight = key.split("/")
        det = json.loads((links_dir / f"{farm}__{flight}.json").read_text())
        paths = [root / farm / "JPGImages" / flight / n for n in det["images"]]
        boxes = [np.asarray(b, float).reshape(-1, 4) for b in det["boxes"]]
        cache: dict[int, Image.Image] = {}

        def img(i: int, cache: dict = cache, paths: list = paths) -> Image.Image:
            if i not in cache:
                cache[i] = load_rgb(paths[i])
            return cache[i]

        for ci, comp in enumerate(c for c in det["components"] if len(c) >= 2):
            pics = []
            for n, (i, k) in enumerate(comp):
                b = boxes[i][k]
                side = max(3 * max(b[2] - b[0], b[3] - b[1]), 120)
                c, x0, y0 = crop_around(img(i), (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, side)
                draw_boxes(c, b[None], x0, y0, side, "#e4572e")
                name = f"pics/{farm}__{flight}__c{ci}_{n}.jpg"
                c.save(out / name, quality=88)
                pics.append({"file": name, "member": [i, k]})
            tasks.append({"id": f"{key}#c{ci}", "kind": "same", "flight": key, "pics": pics})
        H = {(p["i"], p["j"]): np.asarray(p["H_ab"]) for p in det["pairs"] if p.get("H_ab")}
        for si, comp in enumerate(c for c in det["components"] if len(c) == 1):
            i, k = comp[0]
            b = boxes[i][k]
            cxy = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]])
            for (a, bb), Hab in H.items():
                if i not in (a, bb):
                    continue
                j = bb if i == a else a
                # H maps image b's pixels to image a's: invert when i is b's side.
                M = np.linalg.inv(Hab) if i == a else Hab
                p = apply_homography(M, cxy)[0]
                w, h = img(j).size
                if not (0 <= p[0] < w and 0 <= p[1] < h):
                    continue
                side = max(4 * max(b[2] - b[0], b[3] - b[1]), 160)
                left, x0, y0 = crop_around(img(i), *cxy[0], side)
                draw_boxes(left, b[None], x0, y0, side, "#e4572e")
                right, x1, y1 = crop_around(img(j), p[0], p[1], side)
                draw_boxes(right, boxes[j], x1, y1, side, "#2e86e4")
                d = ImageDraw.Draw(right)
                m = THUMB / 2
                d.line([m - 12, m, m + 12, m], fill="#ffd400", width=2)
                d.line([m, m - 12, m, m + 12], fill="#ffd400", width=2)
                ln, rn = (
                    f"pics/{farm}__{flight}__s{si}_L.jpg",
                    f"pics/{farm}__{flight}__s{si}_R.jpg",
                )
                left.save(out / ln, quality=88)
                right.save(out / rn, quality=88)
                # nearest labelled box in j to the predicted place, in case the answer is "boxed"
                cj = np.column_stack(
                    [(boxes[j][:, 0] + boxes[j][:, 2]) / 2, (boxes[j][:, 1] + boxes[j][:, 3]) / 2]
                )
                nearest = int(np.argmin(np.linalg.norm(cj - p, axis=1))) if len(cj) else None
                tasks.append(
                    {
                        "id": f"{key}#s{si}",
                        "kind": "missed",
                        "flight": key,
                        "left": ln,
                        "right": rn,
                        "member": [i, k],
                        "other": [j, nearest],
                    }
                )
                break  # one check per lone animal is enough
        cache.clear()
    html = (REPO_ROOT / "scripts" / "assets" / "link_review.html").read_text(encoding="utf-8")
    if args.round > 1:  # own saved progress and own file name, so rounds never mix
        html = html.replace('"link-answers-v1"', f'"link-answers-round{args.round}"')
        html = html.replace('"link_answers.json"', f'"link_answers_{args.round}.json"')
    (out / "index.html").write_text(html.replace("__TASKS__", json.dumps(tasks)), encoding="utf-8")
    (out / "tasks.json").write_text(json.dumps({"flights": chosen, "tasks": tasks}, indent=1))
    n_same = sum(t["kind"] == "same" for t in tasks)
    print(
        f"flights: {chosen}; {n_same} same-animal checks, {len(tasks) - n_same} missed-link checks"
    )
    print("open", out / "index.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
