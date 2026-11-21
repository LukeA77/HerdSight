"""Make the review page for images labelled empty where a Phase 1 model is confident it sees
cattle (Phase 2 plan, step C4).

Usage: python scripts/make_empty_review.py

Lists every cross-validation image whose label file is empty and where any Phase 1 D-FINE model
scored a detection at MIN_SCORE or more. For each, writes a reduced copy of the whole photo with
those detections outlined, and close-ups of the strongest ones at full resolution. Output in
$HERDSIGHT_STORAGE/exports/empty_review/:
  index.html   the review page (open in a browser; no internet needed)
  img/*.jpg    the pictures
The reviewer answers "cattle", "no cattle" or "not sure" for each photo and downloads
empty_review_answers.json. Images answered "cattle" are left out of Phase 2 training. Scoring is
unchanged (D-32), so this never alters any reported result.
"""

from __future__ import annotations

import html
import json
import sys

import cv2
import numpy as np

from herdsight.data.icaerus import default_root, iter_images
from herdsight.utils.config import REPO_ROOT
from herdsight.utils.paths import storage_path

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from score_baseline import load_preds  # noqa: E402

MIN_SCORE = 0.9
OVERVIEW_PX = 1400
MAX_CROPS = 4


def main() -> int:
    sp = json.loads((REPO_ROOT / "results" / "splits.json").read_text())["icaerus_v2"]
    folds = [f"fold{k}" for k in range(sp["n_folds"])]
    preds: dict = {}
    for f in folds:
        preds |= load_preds(storage_path("runs", "baselines", "dfine") / f"{f}.jsonl")
    out = storage_path("exports") / "empty_review"
    (out / "img").mkdir(parents=True, exist_ok=True)
    tasks = []
    for s, problem in iter_images(default_root(), with_metadata=False):
        k = f"{s.farm}/{s.flight}/{s.path.name}"
        if problem or not s.labelled or sp["flights"].get(f"{s.farm}/{s.flight}") not in folds:
            continue
        if len(s.boxes_xyxy) or k not in preds:
            continue
        boxes, scores = preds[k]
        sel = np.nonzero(scores >= MIN_SCORE)[0]
        if not sel.size:
            continue
        sel = sel[np.argsort(-scores[sel])]
        img = cv2.imread(str(s.path))
        h, w = img.shape[:2]
        f = OVERVIEW_PX / w
        ov = cv2.resize(img, (OVERVIEW_PX, int(h * f)), interpolation=cv2.INTER_AREA)
        for b in boxes[sel]:
            x0, y0, x1, y1 = (int(v * f) for v in b)
            cv2.rectangle(ov, (x0 - 4, y0 - 4), (x1 + 4, y1 + 4), (0, 0, 255), 2)
        tid = f"t{len(tasks):03d}"
        cv2.imwrite(str(out / "img" / f"{tid}_all.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 85])
        crops = []
        for j, b in enumerate(boxes[sel][:MAX_CROPS]):
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            r = int(max(b[2] - b[0], b[3] - b[1]) * 2.5) + 40
            x0, y0 = max(int(cx - r), 0), max(int(cy - r), 0)
            c = img[y0 : min(int(cy + r), h), x0 : min(int(cx + r), w)].copy()
            bx0, by0, bx1, by1 = (int(b[0] - x0), int(b[1] - y0), int(b[2] - x0), int(b[3] - y0))
            cv2.rectangle(c, (bx0, by0), (bx1, by1), (0, 0, 255), 2)
            c = cv2.resize(c, (300, 300), interpolation=cv2.INTER_AREA)
            name = f"{tid}_c{j}.jpg"
            cv2.imwrite(str(out / "img" / name), c, [cv2.IMWRITE_JPEG_QUALITY, 90])
            crops.append(name)
        tasks.append({"id": tid, "image": k, "n": int(sel.size), "crops": crops})
    rows = []
    for t in tasks:
        crops = "".join(f'<img class="crop" src="img/{c}">' for c in t["crops"])
        rows.append(
            f'<section id="{t["id"]}"><h2>{t["id"]}: {html.escape(t["image"])} '
            f'({t["n"]} outlined)</h2><img class="all" src="img/{t["id"]}_all.jpg">'
            f'<div>{crops}</div><div class="ans">'
            + "".join(
                f'<label><input type="radio" name="{t["id"]}" value="{v}"> {txt}</label>'
                for v, txt in (
                    ("cattle", "Cattle present"),
                    ("no_cattle", "No cattle"),
                    ("unsure", "Not sure"),
                )
            )
            + "</div></section>"
        )
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Empty-image review</title>
<style>body{{font-family:sans-serif;max-width:1450px;margin:auto;padding:12px}}
section{{border-top:2px solid #999;padding:10px 0}} img.all{{width:100%}} img.crop{{margin:4px}}
.ans label{{margin-right:24px;font-size:18px}} #bar{{position:sticky;top:0;background:#fff;
padding:8px;border-bottom:1px solid #ccc}}</style></head><body>
<div id="bar"><b>Is there at least one cow, calf or other cattle in this photo?</b>
Red outlines are where a computer model thought it saw cattle; it is often wrong. Look at the
close-ups and the whole photo. <span id="count"></span>
<button onclick="save()">Download answers</button></div>
{"".join(rows)}
<script>
const ids={json.dumps([t["id"] for t in tasks])};
const images={json.dumps({t["id"]: t["image"] for t in tasks})};
function answers(){{const a={{}};for(const i of ids){{const r=document.querySelector(
'input[name="'+i+'"]:checked');if(r)a[i]={{image:images[i],answer:r.value}};}}return a;}}
function upd(){{document.getElementById('count').textContent=Object.keys(answers()).length+
' of '+ids.length+' answered.';}}
document.addEventListener('change',upd);upd();
function save(){{const b=new Blob([JSON.stringify({{answers:answers(),
saved:new Date().toISOString()}},null,1)],{{type:'application/json'}});const a=
document.createElement('a');a.href=URL.createObjectURL(b);a.download=
'empty_review_answers.json';a.click();}}
</script></body></html>"""
    (out / "index.html").write_text(page, encoding="utf-8")
    print(f"{len(tasks)} images written to {out}; open {out / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
