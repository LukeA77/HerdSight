"""Photos for the README: real drone photos with the detector's marks drawn on them.

Usage: python scripts/make_readme_photos.py

Draws on ICAERUS photos (CC BY 4.0, Institut de l'Elevage) and writes JPEGs to docs/figures/:
* photo_farm.jpg            the whole of one held-back test-farm photo, as the drone saw it;
* photo_detections.jpg      the densest part of the same photo with the final detector's marks from the
                            single Phase 7 run: green = a cow it found, orange ring = a cow it missed,
                            red = a mark where there was no cow;
* photo_crowded.jpg         a close-up of the test-farm photo where the detector did worst, a tight
                            herd (section 3 of the README);
* photo_overlap.jpg         two overlapping photos from one hand-checked flight with the same
                            animals joined by lines, as counting across photos does (section 4).
Photos are chosen by fixed rules from the stored results (written below), never by hand. Nothing
is re-run: the marks are the stored predictions of the one test run
($HERDSIGHT_STORAGE/runs/phase7/home_base.jsonl) at the threshold recorded with it. Writes the
counts shown in each picture to results/readme_photos.json. Public data, not field results.
"""

from __future__ import annotations

import json
import sys

import cv2
import numpy as np

from herdsight.count.truth import verified_identities
from herdsight.data.icaerus import default_root, iter_images
from herdsight.evaluate.centre_match import box_centres, match_hungarian, short_side
from herdsight.utils.config import REPO_ROOT, load_config
from herdsight.utils.paths import storage_path

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from score_baseline import load_preds  # noqa: E402

# ruff: noqa: E501

FIG = REPO_ROOT / "docs" / "figures"
GREEN, ORANGE, RED, WHITE, DARK = (
    (90, 170, 60),
    (40, 140, 240),
    (50, 50, 220),
    (255, 255, 255),
    (30, 30, 30),
)
MIN_ANIMALS = 20  # photos with a real herd in them


def match(t: np.ndarray, b: np.ndarray, kappa: float, sim: float) -> tuple[np.ndarray, np.ndarray]:
    """found[i] for each true animal, hit[j] for each mark."""
    found, hit = np.zeros(len(t), bool), np.zeros(len(b), bool)
    if len(t) and len(b):
        m = match_hungarian(
            box_centres(b), box_centres(t), np.maximum(short_side(t), 1.0), kappa, sim
        )
        found[m.pairs[:, 1]] = True
        hit[m.pairs[:, 0]] = True
    return found, hit


def draw(img: np.ndarray, t: np.ndarray, b: np.ndarray, found: np.ndarray, hit: np.ndarray) -> None:
    w = 4
    for box, ok in zip(b, hit, strict=True):
        x0, y0, x1, y1 = (int(v) for v in box)
        cv2.rectangle(img, (x0, y0), (x1, y1), GREEN if ok else RED, w)
    for box, ok in zip(t, found, strict=True):
        if not ok:
            cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
            r = int(max(box[2] - box[0], box[3] - box[1]) * 0.5)
            cv2.circle(img, (cx, cy), r, ORANGE, w)


def densest_window(img: np.ndarray, pts: np.ndarray, cw: int = 1800, ch: int = 1012) -> tuple:
    """The 16:9 window holding the most animals (grid search), so the cattle are big enough to see."""
    h, w = img.shape[:2]
    cw, ch = min(cw, w), min(ch, h)
    best = (-1, 0, 0)
    for top in range(0, h - ch + 1, 50):
        for left in range(0, w - cw + 1, 50):
            n = int(in_crop(pts, left, top, cw, ch).sum())
            if n > best[0]:
                best = (n, left, top)
    return best[1], best[2], cw, ch


def in_crop(c: np.ndarray, left: int, top: int, cw: int, ch: int) -> np.ndarray:
    return (c[:, 0] >= left) & (c[:, 0] < left + cw) & (c[:, 1] >= top) & (c[:, 1] < top + ch)


def legend(img: np.ndarray, items: list[tuple[str, tuple]], note: str) -> np.ndarray:
    """A plain strip under the photo with the key and a one-line note."""
    h, w = img.shape[:2]
    s = w / 1400
    strip = np.full((int(56 * s), w, 3), 250, np.uint8)
    x = int(20 * s)
    y = int(36 * s)
    for label, col in items:
        cv2.rectangle(strip, (x, y - int(18 * s)), (x + int(22 * s), y + int(4 * s)), col, -1)
        cv2.putText(
            strip,
            label,
            (x + int(30 * s), y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62 * s,
            DARK,
            max(1, int(1.6 * s)),
            cv2.LINE_AA,
        )
        x += int((44 + 11 * len(label)) * s)
    cv2.putText(
        strip,
        note,
        (x + int(20 * s), y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55 * s,
        (90, 90, 90),
        max(1, int(1.4 * s)),
        cv2.LINE_AA,
    )
    return np.vstack([img, strip])


def save(img: np.ndarray, name: str, width: int = 1400) -> None:
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(FIG / name), img, [cv2.IMWRITE_JPEG_QUALITY, 82])


def test_farm_photos(out: dict, kappa: float, sim: float) -> None:
    res = json.loads((REPO_ROOT / "results" / "phase7" / "test_farm.json").read_text())
    thr = float(res["calibration"]["home_base"]["threshold"])
    preds = load_preds(storage_path("runs", "phase7") / "home_base.jsonl")
    rows = []
    for s, problem in iter_images(default_root(), with_metadata=False):
        k = f"{s.farm}/{s.flight}/{s.path.name}"
        if s.farm != res["test_farm"] or problem is not None or k not in preds:
            continue
        t = np.asarray(s.boxes_xyxy, float).reshape(-1, 4)
        if len(t) < MIN_ANIMALS:
            continue
        b, sc = preds[k]
        b = b[sc >= thr]
        found, hit = match(t, b, kappa, sim)
        f1 = 2 * found.sum() / max(len(t) + len(b), 1)
        rows.append((f1, k, s.path, t, b, found, hit))
    rows.sort(key=lambda r: r[0])
    # rule: the best photo (highest F1 of found and correct shares) and the worst, both 20+ animals
    for tag, (_f1, k, path, t, b, found, hit) in (("best", rows[-1]), ("worst", rows[0])):
        img = cv2.imread(str(path))
        if tag == "best":
            save(img, "photo_farm.jpg")
        draw(img, t, b, found, hit)
        left, top, cw, ch = densest_window(img, box_centres(t))
        crop = img[top : top + ch, left : left + cw]
        tin = in_crop(box_centres(t), left, top, cw, ch)
        bin_ = in_crop(box_centres(b), left, top, cw, ch) if len(b) else np.zeros(0, bool)
        n_true, n_found = int(tin.sum()), int(found[tin].sum())
        n_false = int((~hit[bin_]).sum())
        note = f"In this close-up: {n_found} of {n_true} cattle found, {n_false} marks in the wrong place"
        crop = legend(
            crop, [("found", GREEN), ("missed", ORANGE), ("mark in the wrong place", RED)], note
        )
        name = "photo_detections.jpg" if tag == "best" else "photo_crowded.jpg"
        save(crop, name)
        out[tag] = {
            "image": k,
            "animals_in_photo": len(t),
            "found_in_photo": int(found.sum()),
            "marks_in_photo": len(b),
            "close_up": {"animals": n_true, "found": n_found, "wrong_marks": n_false},
            "file": f"docs/figures/{name}",
        }


def overlap_photo(out: dict) -> None:
    """Two overlapping stills of a hand-checked flight, with the same animals joined."""
    truth = json.loads((REPO_ROOT / "results" / "data" / "s3_truth.json").read_text())["flights"]
    links = storage_path("processed", "s3_links")
    best = None
    for key in sorted(truth):
        farm, flight = key.split("/")
        path = links / f"{farm}__{flight}.json"
        if not path.exists():
            continue
        ln = json.loads(path.read_text(encoding="utf-8"))
        try:  # the hand-checked animal identities, not the automatic candidate links
            groups = verified_identities(key, ln).groups
        except KeyError:
            continue
        comp_of = {}
        for c, members in enumerate(groups):
            for im, bx in members:
                comp_of[(int(im), int(bx))] = c
        for p in ln["pairs"]:
            i, j = int(p["i"]), int(p["j"])
            ni = [bx for (im, bx) in comp_of if im == i]
            shared = {comp_of[(i, bx)] for bx in ni} & {
                c for (im, _), c in comp_of.items() if im == j
            }
            # rule: the pair of stills sharing the most animals, across all hand-checked flights
            if best is None or len(shared) > best[0]:
                best = (len(shared), key, ln, i, j, comp_of)
    if best is None:
        print("no linked flights found; overlap photo skipped")
        return
    n_shared, key, ln, i, j, comp_of = best
    farm, flight = key.split("/")
    root = default_root()
    paths = {}
    for s, _ in iter_images(root, with_metadata=False):
        if (
            s.farm == farm
            and s.flight == flight
            and s.path.name in (ln["images"][i], ln["images"][j])
        ):
            paths[s.path.name] = s.path
    a = cv2.imread(str(paths[ln["images"][i]]))
    b = cv2.imread(str(paths[ln["images"][j]]))
    scale = 1400 / (a.shape[1] + b.shape[1])
    a = cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gap = 12
    canvas = np.full((max(a.shape[0], b.shape[0]), a.shape[1] + gap + b.shape[1], 3), 255, np.uint8)
    canvas[: a.shape[0], : a.shape[1]] = a
    canvas[: b.shape[0], a.shape[1] + gap :] = b
    boxes_a = np.asarray(ln["boxes"][i], float).reshape(-1, 4) * scale
    boxes_b = np.asarray(ln["boxes"][j], float).reshape(-1, 4) * scale
    ca, cb = box_centres(boxes_a), box_centres(boxes_b) + [a.shape[1] + gap, 0]
    where_b = {comp_of[(j, k)]: k for k in range(len(boxes_b)) if (j, k) in comp_of}
    for k in range(len(boxes_a)):
        c = comp_of.get((i, k))
        if c is not None and c in where_b:
            cv2.line(
                canvas,
                tuple(int(v) for v in ca[k]),
                tuple(int(v) for v in cb[where_b[c]]),
                (60, 200, 250),
                1,
                cv2.LINE_AA,
            )
    for pts in (ca, cb):
        for p in pts:
            cv2.circle(canvas, tuple(int(v) for v in p), 5, GREEN, 2, cv2.LINE_AA)
    total = len(boxes_a) + len(boxes_b)
    note = f"Adding up both photos gives {total}. Joining the {n_shared} cows seen twice leaves {total - n_shared}."
    canvas = legend(canvas, [("cow", GREEN), ("same cow", (60, 200, 250))], note)
    save(canvas, "photo_overlap.jpg")
    out["overlap"] = {
        "flight": key,
        "images": [ln["images"][i], ln["images"][j]],
        "animals_in_both_photos": total,
        "same_animal_pairs": n_shared,
        "different_animals": total - n_shared,
        "file": "docs/figures/photo_overlap.jpg",
    }


def main() -> int:
    ev = load_config("evaluation")
    kappa, sim = float(ev.require("centre_kappa")), float(ev.require("centre_match_similarity"))
    out: dict = {
        "note": "Real ICAERUS photos (CC BY 4.0) with the stored marks of the single test run "
        "drawn on; public data, not field results.",
    }
    test_farm_photos(out, kappa, sim)
    overlap_photo(out)
    (REPO_ROOT / "results" / "readme_photos.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8"
    )
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
