"""Apply the C6 rules to the SAM results and make the review page (Phase 2, step C6).

Usage: python scripts/finalise_sam_labels.py --labels <labels.jsonl>

1. Scores SAM's box centres against the published upright labels (centre-matched recall). Below
   the frozen minimum, every oriented box keeps its upright centre.
2. Writes the training labels to $HERDSIGHT_STORAGE/runs/labels/sam_obb_v1/final_labels.jsonl and
   a summary to results/phase2/c6_sam_labels.json.
3. Writes the review page for 100 labels, stratified by fold and flag, to
   $HERDSIGHT_STORAGE/exports/sam_review/index.html. It works without internet; the reviewer
   downloads sam_review_answers.json.

Oriented training stays paused until the reviewed share reaches the frozen acceptance rate.
"""

# ruff: noqa: E501  (HTML page)

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from herdsight.data.icaerus import default_root, iter_images
from herdsight.evaluate.centre_match import box_centres, match_hungarian, short_side
from herdsight.labels.sam_final import final_obb, select_for_review
from herdsight.utils.config import REPO_ROOT, load_config
from herdsight.utils.io import load_rgb
from herdsight.utils.paths import storage_path

CROP_PX = 340
N_REVIEW = 100


def draw_crop(rgb: np.ndarray, box: list[float], obb: np.ndarray) -> np.ndarray:
    """A close-up with the upright label in white and the training label in green."""
    x0, y0, x1, y1 = box
    side = max(4 * max(x1 - x0, y1 - y0), 120)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left, top = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
    crop = rgb[top : int(top + side), left : int(left + side)]
    scale = CROP_PX / max(crop.shape[:2])
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    a = (int((x0 - left) * scale), int((y0 - top) * scale))
    b = (int((x1 - left) * scale), int((y1 - top) * scale))
    cv2.rectangle(big, a, b, (255, 255, 255), 1)
    rect = (
        ((obb[0] - left) * scale, (obb[1] - top) * scale),
        (obb[2] * scale, obb[3] * scale),
        float(np.degrees(obb[4])),
    )
    pts = cv2.boxPoints(rect).astype(np.int32)
    cv2.polylines(big, [pts], True, (40, 230, 70), 2)
    return big


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Rotated Box Check</title>
<style>
body{font-family:Arial,sans-serif;margin:0;background:#f4f5f6;color:#1d2730}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #d5dbe0;padding:12px 16px;z-index:2}
h1{font-size:20px;margin:0 0 4px}p{margin:4px 0;font-size:14px;line-height:1.4}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px;padding:16px}
.c{background:#fff;border:2px solid #d5dbe0;border-radius:8px;padding:8px}
.c img{width:100%;display:block;border-radius:4px}.n{font-size:12px;color:#5a6670;margin:0 0 5px}
.b{display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px;margin-top:7px}
button{font:inherit;font-size:14px;padding:9px 4px;border:1px solid #9aa5ae;border-radius:5px;background:#fff;cursor:pointer}
.good{border-color:#28803f}.bad{border-color:#b53324}.unsure{border-color:#a97900}
button.on{color:#fff}.good.on{background:#28803f}.bad.on{background:#b53324}.unsure.on{background:#a97900}
#save{background:#1f3a4d;color:#fff;border:0;padding:10px 14px;margin-top:6px}
</style></head><body><header><h1>Rotated box check</h1>
<p><b>Good</b>: the green box follows the marked animal's body and direction reasonably well. <b>Bad</b>: it covers a different animal, a shadow or several animals, or points clearly the wrong way. The white box is the original label and shows which animal to judge.</p>
<p><span id="done">0</span> of __N__ answered. Answers save in this browser. <button id="save">Download answers</button></p></header>
<main id="grid">__CARDS__</main><script>
const KEY='herdsight_sam_review_v1';let a={};try{a=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
function draw(){document.querySelectorAll('.c').forEach(c=>{c.querySelectorAll('button').forEach(b=>b.classList.toggle('on',a[c.dataset.id]===b.dataset.v))});document.getElementById('done').textContent=Object.keys(a).length}
document.getElementById('grid').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;a[b.closest('.c').dataset.id]=b.dataset.v;try{localStorage.setItem(KEY,JSON.stringify(a))}catch(x){}draw()});
document.getElementById('save').onclick=()=>{const u=URL.createObjectURL(new Blob([JSON.stringify(a,null,1)],{type:'application/json'}));const l=document.createElement('a');l.href=u;l.download='sam_review_answers.json';l.click()};draw();
</script></body></html>"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    args = ap.parse_args(argv)
    chk = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/freeze.py"),
            "--phase",
            "phase2_prereg",
            "--check",
        ],
        capture_output=True,
        text=True,
    )
    print(chk.stdout.strip())
    if chk.returncode != 0:
        return 1
    rows = [
        json.loads(x) for x in args.labels.read_text(encoding="utf-8").splitlines() if x.strip()
    ]
    sp = json.loads((REPO_ROOT / "results" / "splits.json").read_text())["icaerus_v2"]
    allowed = {f"fold{k}" for k in range(sp["n_folds"])}
    outside = [r["image"] for r in rows if r["fold"] not in allowed]
    if outside:
        raise SystemExit(f"{len(outside)} images outside the folds, e.g. {outside[0]}")
    ev, p2 = load_config("evaluation"), load_config("phase2")
    kappa, sim = float(ev.require("centre_kappa")), float(ev.require("centre_match_similarity"))
    min_recall = float(p2.require("sam_centre_recall_min"))
    tp = total = 0
    for r in rows:
        truth = np.asarray(r["boxes"], float).reshape(-1, 4)
        pred = np.array([x["obb"][:2] for x in r["labels"] if x["obb"] is not None], float)
        total += len(truth)
        if len(pred):
            tp += match_hungarian(
                pred.reshape(-1, 2),
                box_centres(truth),
                np.maximum(short_side(truth), 1.0),
                kappa,
                sim,
            ).tp
    recall = tp / max(total, 1)
    keep_centre = recall < min_recall
    run_dir = storage_path("runs", "labels", "sam_obb_v1", create=True)
    final = {}
    with (run_dir / "final_labels.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            obbs = [
                final_obb(b, x, keep_centre) for b, x in zip(r["boxes"], r["labels"], strict=True)
            ]
            final[r["image"]] = obbs
            line = {
                "image": r["image"],
                "fold": r["fold"],
                "obb": [[round(float(v), 4) for v in o] for o in obbs],
            }
            fh.write(json.dumps(line) + "\n")
    labels = [x for r in rows for x in r["labels"]]
    flags = Counter(f for x in labels for f in x["flags"])
    fallback = sum(x["obb"] is None or bool(x["flags"]) for x in labels)
    summary = {
        "note": "Training labels only; never used to score a model. Public data.",
        "images": len(rows),
        "animals": total,
        "sam_centre_recall": recall,
        "sam_centre_recall_min": min_recall,
        "centres_kept_from_upright_labels": keep_centre,
        "labels_with_a_flag_using_angle_0_fallback": fallback,
        "flag_counts": dict(sorted(flags.items())),
        "no_mask": sum(x["obb"] is None for x in labels),
        "eye_check": "pending",
    }
    (REPO_ROOT / "results/phase2/c6_sam_labels.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8"
    )
    picks = select_for_review(rows, N_REVIEW)
    by_image = {r["image"]: r for r in rows}
    paths = {
        f"{s.farm}/{s.flight}/{s.path.name}": s.path
        for s, _ in iter_images(default_root(), with_metadata=False)
    }
    out = storage_path("exports") / "sam_review"
    (out / "img").mkdir(parents=True, exist_ok=True)
    cards, key = [], {}
    needed: dict[str, list[tuple[int, int]]] = {}
    for n, (image, k) in enumerate(picks, 1):
        needed.setdefault(image, []).append((n, k))
    for image, items in needed.items():
        rgb = np.asarray(load_rgb(paths[image]))
        for n, k in items:
            crop = draw_crop(rgb, by_image[image]["boxes"][k], final[image][k])
            cv2.imwrite(
                str(out / "img" / f"{n:03d}.jpg"),
                cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 88],
            )
            x = by_image[image]["labels"][k]
            key[f"{n:03d}"] = {
                "image": image,
                "box": k,
                "fold": by_image[image]["fold"],
                "flagged": x["obb"] is None or bool(x["flags"]),
            }
    for n in range(1, len(picks) + 1):
        i = f"{n:03d}"
        cards.append(
            f'<section class="c" data-id="{html.escape(i)}"><p class="n">Label {n}</p><img loading="lazy" src="img/{i}.jpg" alt="Animal {n} with its rotated box">'
            '<div class="b"><button class="good" data-v="good">Good</button><button class="bad" data-v="bad">Bad</button><button class="unsure" data-v="unsure">Not sure</button></div></section>'
        )
    (out / "index.html").write_text(
        PAGE.replace("__N__", str(len(picks))).replace("__CARDS__", "".join(cards)),
        encoding="utf-8",
    )
    (out / "review_key.json").write_text(json.dumps(key, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"review page: {out / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
