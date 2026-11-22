"""Nested hard-negative mining, home-base part (Phase 2 plan step G3; D-34).

Usage:
  python scripts/mine_hard_negatives.py mine --models DIR [--workers N] [--limit-images N]
  python scripts/mine_hard_negatives.py select
  python scripts/mine_hard_negatives.py apply --answers FILE
  python scripts/mine_hard_negatives.py package

DIR holds the three mining models from gpu/runpod/job_g3.sh: g3_mine_fold<k>/last.onnx, each
trained on fold k's tiles only. Everything is written under
$HERDSIGHT_STORAGE/runs/g3_mining/ and $HERDSIGHT_STORAGE/exports/hardneg_review/.

mine    Cuts every cross-validation image into the same 640 px tiles as the training package
        (configs/baselines.yaml) and keeps the tiles that contain no part of any labelled animal.
        Each such tile is run through the two mining models that did not train on its fold. A
        tile is a candidate for a model when that model's strongest detection scores at least
        SCORE_MIN. Images on the unlabelled-cattle list (C4) and the test farm are never used.
        Resumable: images already done are skipped.
select  For each fold k (the model that trains on the other two folds a and b): the hard
        negatives are tiles of fold b found by the model trained on a, and tiles of fold a found
        by the model trained on b, half from each, ranked by the strongest detection, at most
        PER_IMAGE tiles from any one photo. The number needed is hard_negative_ratio
        (configs/phase2.yaml) times the animal tiles in train_fold<k>; OVERSELECT more are taken
        so tiles the reviewer rejects can be dropped. Writes the review page.
apply   Reads the reviewer's answers (tiles marked as showing an animal are dropped), keeps the
        best-ranked accepted tiles up to the number needed, and writes
        processed/herdsight_hardneg_v1/: images/*.jpg and hardneg_fold<k>.json (COCO, no boxes),
        for G2 to add to train_fold<k>.json. Tiles from photos on the training exclusions list
        (C4, which can grow after mining) are dropped as well.
package Zips the set with a manifest into exports/herdsight_hardneg_v1.zip (uploaded once as a
        private Kaggle dataset, like the tile package) and writes its manifest to
        results/phase2/.
Speed (laptop CPU, about 0.25 s per tile per model): mining uses the same 640 px tiles but
without overlap (overlap only adds near-duplicate candidates, which PER_IMAGE then drops), and a
random IMAGE_SHARE of the photos (seeded), which gives several times the candidates needed; if
`select` reports a shortfall, `mine --image-share 1` mines the rest (resumable).
These settings were fixed on 2026-11-22, before any mining model was trained.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import sys
import zipfile
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from herdsight.data.icaerus import default_root, index_farms, iter_images
from herdsight.detect.dfine_onnx import CATTLE_LABEL, preprocess
from herdsight.detect.tiling import Tile, tile_grid
from herdsight.utils.config import REPO_ROOT, load_config
from herdsight.utils.io import load_rgb
from herdsight.utils.paths import storage_path

SCORE_MIN = 0.3  # strongest detection on an animal-free tile, to be a candidate
PER_IMAGE = 2  # at most this many tiles from one photo, per fold
OVERSELECT = 1.15  # spare tiles for those the reviewer rejects
MAX_BOXES = 5  # detections kept per candidate (for the review page)
THREADS = 2  # ONNX Runtime threads per worker process
IMAGE_SHARE = 0.5  # share of photos mined (seeded)
IMAGE_SEED = 20261122
MINING_OVERLAP_PX = 0  # tiles without overlap
NAME = "herdsight_hardneg_v1"

_SESSIONS: dict[str, object] = {}


def folds_and_groups() -> tuple[list[str], dict[str, str]]:
    sp = json.loads((REPO_ROOT / "results" / "splits.json").read_text(encoding="utf-8"))
    sp = sp["icaerus_v2"]
    return [f"fold{k}" for k in range(sp["n_folds"])], sp["flights"]


def animal_free(tile: Tile, boxes: np.ndarray) -> bool:
    """True if no labelled box overlaps the tile at all (even partly)."""
    if not len(boxes):
        return True
    b = np.asarray(boxes, float)
    ix = np.minimum(b[:, 2], tile.x1) - np.maximum(b[:, 0], tile.x0)
    iy = np.minimum(b[:, 3], tile.y1) - np.maximum(b[:, 1], tile.y0)
    return not np.any((ix > 0) & (iy > 0))


def _init(model_paths: dict[str, str]) -> None:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = THREADS
    opts.inter_op_num_threads = 1
    for fold, p in model_paths.items():
        _SESSIONS[fold] = ort.InferenceSession(p, opts, providers=["CPUExecutionProvider"])


def mine_image(job: tuple) -> dict:
    """Candidates from one photo: {key, records: [...]}."""
    path, farm, flight, fold, boxes, tile_px, overlap = job
    im = load_rgb(Path(path))
    out = []
    for t in tile_grid(im.width, im.height, tile_px, overlap):
        if not animal_free(t, boxes):
            continue
        crop = im.crop((t.x0, t.y0, t.x1, t.y1))
        x, sizes = preprocess(crop, tile_px)
        for model_fold, sess in _SESSIONS.items():
            if model_fold == fold:
                continue  # never mine a fold with a model trained on it
            labels, bxs, scores = sess.run(None, {"images": x, "orig_target_sizes": sizes})
            keep = labels[0] == CATTLE_LABEL
            s, b = scores[0][keep], bxs[0][keep]
            if not len(s) or s.max() < SCORE_MIN:
                continue
            order = np.argsort(-s)[:MAX_BOXES]
            order = [i for i in order if s[i] >= SCORE_MIN]
            out.append(
                {
                    "tile": f"{farm}__{flight}__{Path(path).stem}__{t.x0}_{t.y0}.jpg",
                    "source": f"{farm}/{flight}/{Path(path).name}",
                    "xyxy": [t.x0, t.y0, t.x1, t.y1],
                    "fold": fold,
                    "model_fold": model_fold,
                    "score": round(float(s.max()), 4),
                    "boxes": [[round(float(v), 1) for v in b[i]] for i in order],
                    "scores": [round(float(s[i]), 4) for i in order],
                }
            )
    return {"key": f"{farm}/{flight}/{Path(path).name}", "records": out}


def image_draw(path: str) -> float:
    """A number in [0, 1) fixed for each photo (from its file name and IMAGE_SEED)."""
    h = hashlib.sha256(f"{IMAGE_SEED}:{Path(path).name}".encode()).hexdigest()
    return int(h[:12], 16) / 16**12


def run_dir() -> Path:
    return storage_path("runs", "g3_mining", create=True)


def cmd_mine(args: argparse.Namespace) -> int:
    folds, flights = folds_and_groups()
    models = {f: args.models / f"g3_mine_{f}" / "last.onnx" for f in folds}
    missing = [str(p) for p in models.values() if not p.exists()]
    if missing:
        raise SystemExit(f"missing mining models: {missing}")
    cfg = load_config("baselines")
    tile_px, overlap = int(cfg.require("dfine_tile_px")), MINING_OVERLAP_PX
    excluded = set(training_exclusions())
    out = run_dir() / "candidates.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(x)["key"] for x in out.read_text(encoding="utf-8").splitlines() if x}
    jobs = []
    for s, problem in iter_images(default_root(), with_metadata=False):
        fold = flights.get(f"{s.farm}/{s.flight}")
        key = f"{s.farm}/{s.flight}/{s.path.name}"
        if fold not in folds or problem or not s.labelled or key in excluded or key in done:
            continue
        boxes = np.asarray(s.boxes_xyxy, float).reshape(-1, 4).tolist()
        jobs.append((str(s.path), s.farm, s.flight, fold, boxes, tile_px, overlap))
    if args.image_share < 1.0:  # a fixed choice per photo, the same on every run
        jobs = [j for j in jobs if image_draw(j[0]) < args.image_share]
    if args.limit_images:
        jobs = jobs[: args.limit_images]
    print(f"{len(jobs)} images to mine ({len(done)} already done)", flush=True)
    paths = {f: str(p) for f, p in models.items()}
    with (
        ProcessPoolExecutor(args.workers, initializer=_init, initargs=(paths,)) as pool,
        out.open("a", encoding="utf-8") as fh,
    ):
        for n, res in enumerate(pool.map(mine_image, jobs, chunksize=1), 1):
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            if n % 25 == 0:
                print(f"{n}/{len(jobs)} images", flush=True)
    return 0


def load_candidates() -> list[dict]:
    path = run_dir() / "candidates.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    return [r for row in rows for r in row["records"]]


def training_exclusions() -> list[str]:
    """Photos left out of Phase 2 training (C4): farm/flight/photo name."""
    return json.loads((REPO_ROOT / "results" / "data" / "training_exclusions.json").read_text())


def needed_per_fold() -> dict[str, int]:
    folds, _ = folds_and_groups()
    ratio = float(load_config("phase2").require("hard_negative_ratio"))
    man = json.loads(
        (storage_path("processed", "herdsight_tiles_v1") / "manifest.json").read_text()
    )["counts"]
    return {
        k: int(round(ratio * (man[k]["train_tiles"] - man[k]["train_empty_tiles"]))) for k in folds
    }


def ranked(cands: list[dict], k: str, folds: list[str]) -> list[tuple[str, list[dict]]]:
    """For fold k: [(source description, candidates ranked)] for its two mining sources."""
    a, b = [f for f in folds if f != k]
    out = []
    for model_fold, tile_fold in ((a, b), (b, a)):
        pool = [c for c in cands if c["model_fold"] == model_fold and c["fold"] == tile_fold]
        pool.sort(key=lambda c: (-c["score"], c["tile"]))
        out.append((f"model {model_fold} on {tile_fold}", pool))
    return out


def take(pool: list[dict], n: int, skip: set[str] = frozenset()) -> list[dict]:
    per_image: dict[str, int] = {}
    picked = []
    for c in pool:
        if len(picked) >= n:
            break
        if c["tile"] in skip or per_image.get(c["source"], 0) >= PER_IMAGE:
            continue
        per_image[c["source"]] = per_image.get(c["source"], 0) + 1
        picked.append(c)
    return picked


def cmd_select(args: argparse.Namespace) -> int:
    folds, _ = folds_and_groups()
    cands = load_candidates()
    need = needed_per_fold()
    selection: dict = {"need": need, "folds": {}}
    review: dict[str, dict] = {}
    for k in folds:
        per_source = math.ceil(need[k] / 2 * OVERSELECT)
        chosen = []
        for name, pool in ranked(cands, k, folds):
            got = take(pool, per_source)
            chosen += got
            print(f"{k}: {name}: {len(pool)} candidates, {len(got)} selected", flush=True)
        selection["folds"][k] = [c["tile"] for c in chosen]
        for c in chosen:
            r = review.setdefault(c["tile"], {**c, "for_folds": []})
            r["for_folds"].append(k)
            if c["score"] > r["score"]:
                r.update({key: c[key] for key in ("score", "boxes", "scores", "model_fold")})
    (run_dir() / "selection.json").write_text(json.dumps(selection, indent=1), encoding="utf-8")
    write_review(sorted(review.values(), key=lambda r: (r["source"], r["xyxy"])))
    return 0


def photo_resolver() -> Callable[[str], Path]:
    """Map a candidate's `source` (farm/flight/photo name) to the photo file on disk."""
    index = index_farms(default_root())

    def resolve(source: str) -> Path:
        farm, _, name = source.split("/")
        return index[farm].images[Path(name).stem]

    return resolve


def write_review(items: list[dict]) -> None:
    resolve = photo_resolver()
    out = storage_path("exports", "hardneg_review", create=True)
    (out / "img").mkdir(exist_ok=True)
    cards = []
    for i, r in enumerate(items):
        name = f"t{i:05d}.jpg"
        im = load_rgb(resolve(r["source"])).crop(tuple(r["xyxy"]))
        d = ImageDraw.Draw(im)
        for b in r["boxes"]:
            d.rectangle(b, outline=(255, 40, 40), width=3)
        im.resize((320, 320), Image.Resampling.LANCZOS).save(out / "img" / name, quality=88)
        cards.append(
            f'<figure data-tile="{html.escape(r["tile"])}"><img src="img/{name}" loading="lazy" '
            f'alt="tile {i + 1}"><figcaption>{i + 1}</figcaption></figure>'
        )
    page = REVIEW_PAGE.replace("{N}", str(len(items))).replace("{CARDS}", "\n".join(cards))
    (out / "index.html").write_text(page, encoding="utf-8")
    print(f"review page: {out / 'index.html'} ({len(items)} tiles)")


REVIEW_PAGE = """<!doctype html>
<html lang="en-GB"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hard-negative check</title>
<style>
body{font:15px/1.45 system-ui,sans-serif;margin:0;background:#f6f5f1;color:#1f1f1e}
header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:12px 16px;
z-index:1}
h1{font-size:18px;margin:0 0 4px}
p{margin:4px 0;max-width:70ch}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:8px;padding:12px}
figure{margin:0;background:#fff;border:3px solid transparent;cursor:pointer;position:relative}
figure img{width:100%;display:block}
figcaption{position:absolute;top:2px;left:4px;font-size:12px;color:#fff;
text-shadow:0 0 3px #000}
figure.animal{border-color:#d62828}
figure.animal::after{content:"ANIMAL";position:absolute;bottom:4px;right:4px;background:#d62828;
color:#fff;font-size:12px;padding:1px 6px}
button{font:inherit;padding:6px 14px;margin-right:8px}
</style></head><body>
<header><h1>Hard-negative check: {N} tiles</h1>
<p>Each tile is a patch of pasture where a model wrongly saw cattle (red boxes). Click any tile
that <b>does</b> show a cow or calf, even partly or in shadow; it turns red. Leave the rest.
Not sure? Mark it as an animal. Your clicks are kept in this browser if you stop halfway.</p>
<p><span id="count">0</span> marked. <button id="save">Download answers</button></p></header>
<main>
{CARDS}
</main>
<script>
const KEY = "hardneg_review_v1";
let marked = new Set();
try { marked = new Set(JSON.parse(localStorage.getItem(KEY) || "[]")); } catch (e) {}
const figs = document.querySelectorAll("figure");
function show() {
  figs.forEach(f => f.classList.toggle("animal", marked.has(f.dataset.tile)));
  document.getElementById("count").textContent = marked.size;
  try { localStorage.setItem(KEY, JSON.stringify([...marked])); } catch (e) {}
}
figs.forEach(f => f.addEventListener("click", () => {
  const t = f.dataset.tile;
  marked.has(t) ? marked.delete(t) : marked.add(t);
  show();
}));
document.getElementById("save").addEventListener("click", () => {
  const body = JSON.stringify({reviewed: figs.length, animal: [...marked].sort()}, null, 1);
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([body], {type: "application/json"}));
  a.download = "hardneg_review_answers.json";
  a.click();
});
show();
</script></body></html>
"""


def cmd_apply(args: argparse.Namespace) -> int:
    folds, _ = folds_and_groups()
    answers = json.loads(args.answers.read_text(encoding="utf-8"))
    cands = load_candidates()
    # tiles marked as showing an animal, and tiles cut from photos left out of training (C4,
    # which can grow after mining) are never used
    excluded_photos = set(training_exclusions())
    animal = set(answers["animal"]) | {c["tile"] for c in cands if c["source"] in excluded_photos}
    need = needed_per_fold()
    sel = json.loads((run_dir() / "selection.json").read_text(encoding="utf-8"))
    reviewed = {t for k in folds for t in sel["folds"][k]}
    out = storage_path("processed", NAME, create=True)
    (out / "images").mkdir(exist_ok=True)
    resolve = photo_resolver()
    summary = {
        "need": need,
        "rejected_as_animal": len(set(answers["animal"]) & reviewed),
        "from_excluded_photos": len((animal - set(answers["animal"])) & reviewed),
        "folds": {},
    }
    for k in folds:
        tiles = []
        for _, pool in ranked(cands, k, folds):
            pool = [c for c in pool if c["tile"] in set(sel["folds"][k])]
            tiles += take(pool, math.ceil(need[k] / 2), skip=animal)
        images = []
        for i, c in enumerate(tiles, 1):
            dest = out / "images" / c["tile"]
            if not dest.exists():
                load_rgb(resolve(c["source"])).crop(tuple(c["xyxy"])).save(dest, quality=95)
            images.append({"id": i, "file_name": c["tile"], "width": 640, "height": 640})
        coco = {"images": images, "annotations": [], "categories": [{"id": 1, "name": "cattle"}]}
        (out / f"hardneg_{k}.json").write_text(json.dumps(coco), encoding="utf-8")
        summary["folds"][k] = {"tiles": len(images), "short_by": max(0, need[k] - len(images))}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


def cmd_package(args: argparse.Namespace) -> int:
    """Zip the reviewed set for the GPU sessions (uploaded once as a private Kaggle dataset)."""
    src = storage_path("processed", NAME)
    summary = json.loads((src / "summary.json").read_text(encoding="utf-8"))
    if any(f["short_by"] for f in summary["folds"].values()):
        raise SystemExit("a fold is short of hard negatives; see summary.json")
    images = sorted((src / "images").glob("*.jpg"))
    manifest = {
        "name": NAME,
        "made_by": "scripts/mine_hard_negatives.py (mine, select, apply, package)",
        "settings": {
            "score_min": SCORE_MIN,
            "per_image": PER_IMAGE,
            "overselect": OVERSELECT,
            "mining_overlap_px": MINING_OVERLAP_PX,
        },
        "training_exclusions_sha256": hashlib.sha256(
            (REPO_ROOT / "results" / "data" / "training_exclusions.json").read_bytes()
        ).hexdigest(),
        "summary": summary,
        "images": len(images),
        "licence": "Crops of ICAERUS grazing cows v2 photos, CC BY 4.0",
    }
    (src / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    zpath = storage_path("exports", create=True) / f"{NAME}.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as z:
        z.write(src / "manifest.json", f"{NAME}/manifest.json")
        for k in summary["folds"]:
            z.write(src / f"hardneg_{k}.json", f"{NAME}/hardneg_{k}.json")
        for im in images:
            z.write(im, f"{NAME}/images/{im.name}")
    h = hashlib.sha256()
    with zpath.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    manifest.update(zip_bytes=zpath.stat().st_size, zip_sha256=h.hexdigest())
    dest = REPO_ROOT / "results" / "phase2" / f"{NAME}_manifest.json"
    dest.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"{zpath} ({manifest['zip_bytes'] / 1e6:.0f} MB, {len(images)} tiles); manifest {dest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mine")
    m.add_argument("--models", type=Path, required=True)
    m.add_argument("--workers", type=int, default=6)
    m.add_argument("--limit-images", type=int, default=0)
    m.add_argument("--image-share", type=float, default=IMAGE_SHARE)
    sub.add_parser("select")
    a = sub.add_parser("apply")
    a.add_argument("--answers", type=Path, required=True)
    sub.add_parser("package")
    args = ap.parse_args(argv)
    return {
        "mine": cmd_mine,
        "select": cmd_select,
        "apply": cmd_apply,
        "package": cmd_package,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
