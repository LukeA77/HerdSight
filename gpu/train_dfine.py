"""Phase 2 D-FINE training run (any GPU machine; used by gpu/runpod/ jobs).

Usage:
  python gpu/train_dfine.py --run NAME --package DIR --fold fold0 [options]

Starts from the authors' COCO weights at the pinned D-FINE commit, as in Phase 1
(gpu/train_dfine_baselines_kaggle.py, which stays unchanged as the Phase 1 record). Differences
that Phase 2 needs, each an option:
  --size s|m|n            model size (plan step G4)
  --epochs N              72 for screening, 220 for the authors' full schedule (G0, D-22)
  --recipe phase1|g2      g2 adds the ops in gpu/dfine_ops.py (dark/shade, shadow, greyscale,
                          blur, 90-degree turns, vertical flip) to Phase 1's augmentation (G2)
  --oversample-touching K repeat each training tile with touching animals K times in total (G2)
  --exclude FILE          JSON list of source images ("farm/flight/image.JPG") left out of
                          training, e.g. images with unlabelled cattle (C4, D-32)
  --holdout-group G       farm-date group removed from training and used as the evaluation set
                          (G0: a training-side hold-out, never the fold's validation images)
  --workers N             data-loading processes (the Phase 1 value of 4 left the GPU idle)
  --seed N
Writes WORK/RUN/: last.pth, last.onnx, log.txt, train_stdout.txt, the config, the label files
used, and run.json (settings, hashes, timings, GPU).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

DFINE_COMMIT = "956d1709314c2c6a4df6f34de232054578a7449f"
WEIGHTS = "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_{}_coco.pth"
AUTHORS_BATCH = {"n": 128, "s": 64, "m": 32}  # the authors' custom-dataset configs
RES, BATCH, NUM_CLASSES = 640, 16, 2
HERE = Path(__file__).resolve().parent

PHASE1_OPS = [
    {"type": "RandomPhotometricDistort", "p": 0.5},
    {"type": "RandomZoomOut", "fill": 0},
    {"type": "RandomIoUCrop", "p": 0.8},
    {"type": "SanitizeBoundingBoxes", "min_size": 1},
    {"type": "RandomHorizontalFlip"},
]
G2_OPS = [
    {"type": "HsRandomRot90", "p": 0.75},
    {"type": "HsRandomVFlip", "p": 0.5},
    {"type": "HsDarkenShade", "p": 0.5},
    {"type": "HsRandomShadow", "p": 0.3},
    {"type": "HsRandomGrayscale", "p": 0.1},
    {"type": "HsRandomBlur", "p": 0.2},
]
TAIL_OPS = [
    {"type": "Resize", "size": [RES, RES]},
    {"type": "SanitizeBoundingBoxes", "min_size": 1},
    {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
    {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
]
# Same export as Phase 1 (gpu/train_dfine_baselines_kaggle.py, EXPORT): 2-image dummy, no grad.
EXPORT = """
import functools, inspect, os, sys, torch
if "dynamo" in inspect.signature(torch.onnx.export).parameters:
    torch.onnx.export = functools.partial(torch.onnx.export, dynamo=False)
path = os.path.abspath("tools/deployment/export_onnx.py")
src = open(path).read()
old = "torch.rand(32, 3, 640, 640)"
assert src.count(old) == 1, "export_onnx.py is not the pinned version"
src = src.replace(old, "torch.rand(2, 3, 640, 640)")
sys.argv = [path] + sys.argv[1:]
torch.set_grad_enabled(False)
exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path})
"""


def sh(*cmd: str, **kw: object) -> None:
    subprocess.run(list(cmd), check=True, **kw)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def source_image(file_name: str) -> tuple[str, str, str]:
    """Tile file name 'farm__flight__stem__x_y.jpg' -> (farm, flight, 'stem.JPG')."""
    farm, flight, stem = file_name.split("__")[:3]
    return farm, flight, f"{stem}.JPG"


def group_of(farm: str, flight: str) -> str:
    sys.path.insert(0, str(HERE.parent / "python"))
    from herdsight.data.splits import FlightStats

    return FlightStats(farm, flight, 0, 0).group


def iou_any(boxes: list[list[float]]) -> bool:
    """Does any pair of COCO [x, y, w, h] boxes overlap?"""
    for i in range(len(boxes)):
        xi, yi, wi, hi = boxes[i]
        for j in range(i + 1, len(boxes)):
            xj, yj, wj, hj = boxes[j]
            if min(xi + wi, xj + wj) > max(xi, xj) and min(yi + hi, yj + hj) > max(yi, yj):
                return True
    return False


def build_sets(args: argparse.Namespace, out: Path) -> tuple[Path, Path, dict]:
    """Training and evaluation label files for this run, and counts for run.json."""
    data = args.package
    coco = json.loads((data / f"train_{args.fold}.json").read_text())
    excluded = set(json.loads(args.exclude.read_text())) if args.exclude else set()
    anns: dict[int, list] = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    train_imgs, hold_imgs, dropped = [], [], 0
    for im in coco["images"]:
        farm, flight, image = source_image(im["file_name"])
        if f"{farm}/{flight}/{image}" in excluded:
            dropped += 1
            continue
        if args.holdout_group and group_of(farm, flight) == args.holdout_group:
            hold_imgs.append(im)
        else:
            train_imgs.append(im)
    if args.holdout_group and not hold_imgs:
        raise SystemExit(f"hold-out group {args.holdout_group} has no tiles in train_{args.fold}")

    def coco_of(images: list[dict], repeat_touching: int) -> tuple[dict, int]:
        out_imgs, out_anns, repeated = [], [], 0
        for im in images:
            a = anns.get(im["id"], [])
            k = repeat_touching if len(a) > 1 and iou_any([x["bbox"] for x in a]) else 1
            repeated += k > 1
            for _ in range(k):
                new_id = len(out_imgs) + 1
                out_imgs.append({**im, "id": new_id})
                for x in a:
                    out_anns.append({**x, "id": len(out_anns) + 1, "image_id": new_id})
        return {
            "images": out_imgs,
            "annotations": out_anns,
            "categories": coco["categories"],
        }, repeated

    train, repeated = coco_of(train_imgs, args.oversample_touching)
    train_path = out / "train.json"
    train_path.write_text(json.dumps(train))
    if args.holdout_group:
        evals, _ = coco_of(hold_imgs, 1)
        eval_path = out / "holdout.json"
        eval_path.write_text(json.dumps(evals))
    else:
        eval_path = data / f"monitor_{args.fold}.json"
    counts = {
        "train_tiles": len(train["images"]),
        "train_tiles_before_repeats": len(train_imgs),
        "tiles_with_touching_animals_repeated": repeated,
        "train_boxes": len(train["annotations"]),
        "excluded_tiles": dropped,
        "holdout_tiles": len(hold_imgs),
    }
    return train_path, eval_path, counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--fold", required=True)
    ap.add_argument("--size", default="s", choices=sorted(AUTHORS_BATCH))
    ap.add_argument("--epochs", type=int, default=72)
    ap.add_argument("--recipe", default="phase1", choices=("phase1", "g2"))
    ap.add_argument("--oversample-touching", type=int, default=1)
    ap.add_argument("--exclude", type=Path, default=None)
    ap.add_argument("--holdout-group", default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work", type=Path, default=Path("/workspace/work"))
    ap.add_argument("--dfine", type=Path, default=Path("/workspace/D-FINE"))
    ap.add_argument(
        "--cpu-smoke",
        type=int,
        default=0,
        help="local check only: N tiles, CPU, given --epochs; never a model to keep",
    )
    args = ap.parse_args(argv)
    args.work, args.package, args.dfine = (
        p.resolve() for p in (args.work, args.package, args.dfine)
    )
    t0 = time.time()
    # augmentation stops for the last epochs: the authors' 220/200, Phase 1's 72/66
    aug_off = {220: 200, 72: 66}.get(args.epochs, max(1, args.epochs - round(args.epochs / 11)))

    repo = args.dfine
    if not repo.exists():
        sh("git", "clone", "-q", "https://github.com/Peterande/D-FINE.git", str(repo))
    sh("git", "-C", str(repo), "checkout", "-q", DFINE_COMMIT)
    if not (repo / ".hs_deps").exists() and not args.cpu_smoke:
        sh(sys.executable, "-m", "pip", "install", "-q", "-r", str(repo / "requirements.txt"))
        sh(
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "onnx",
            "onnxsim",
            "pycocotools",
            "matplotlib",
        )
        sh(sys.executable, "-m", "pip", "install", "-q", "faster-coco-eval==1.8.0")  # see Phase 1
        (repo / ".hs_deps").touch()
    # register the Phase 2 augmentations inside D-FINE (idempotent)
    tdir = repo / "src" / "data" / "transforms"
    shutil.copy2(HERE / "dfine_ops.py", tdir / "hs_ops.py")
    init = tdir / "__init__.py"
    if "hs_ops" not in init.read_text():
        init.write_text(init.read_text() + "\nfrom . import hs_ops  # HerdSight Phase 2 ops\n")

    # fail in seconds, not after downloads and set-up, if a dependency is missing
    sh(sys.executable, "-c", "import src.solver, src.data.transforms.hs_ops", cwd=repo)
    import torch
    import yaml

    if not torch.cuda.is_available() and not args.cpu_smoke:
        raise SystemExit("no GPU")
    out = args.work / args.run
    out.mkdir(parents=True, exist_ok=True)
    weights = args.work / f"dfine_{args.size}_coco.pth"
    if not weights.exists():
        urllib.request.urlretrieve(WEIGHTS.format(args.size), weights.with_suffix(".part"))
        os.replace(weights.with_suffix(".part"), weights)
    train_ann, eval_ann, counts = build_sets(args, out)
    if args.cpu_smoke:  # keep a few tiles with animals in each set
        for p in (train_ann, eval_ann):
            c = json.loads(p.read_text())
            keep = sorted({a["image_id"] for a in c["annotations"]})[: args.cpu_smoke]
            c["images"] = [i for i in c["images"] if i["id"] in keep]
            c["annotations"] = [a for a in c["annotations"] if a["image_id"] in keep]
            p = out / ("smoke_" + p.name)
            p.write_text(json.dumps(c))
        train_ann, eval_ann = out / ("smoke_" + train_ann.name), out / ("smoke_" + eval_ann.name)

    base_cfg = repo / f"configs/dfine/custom/dfine_hgnetv2_{args.size}_custom.yml"
    lr_scale = BATCH / AUTHORS_BATCH[args.size]
    ops = PHASE1_OPS + (G2_OPS if args.recipe == "g2" else []) + TAIL_OPS
    stop_ops = ["RandomPhotometricDistort", "RandomZoomOut", "RandomIoUCrop"] + (
        [o["type"] for o in G2_OPS] if args.recipe == "g2" else []
    )
    img_folder = (args.package / "images").as_posix()
    cfg = {
        "__include__": [base_cfg.as_posix()],
        "output_dir": out.as_posix(),
        "num_classes": NUM_CLASSES,
        "remap_mscoco_category": False,
        "eval_spatial_size": [RES, RES],
        "epochs": args.epochs,
        "train_dataloader": {
            "total_batch_size": 2 if args.cpu_smoke else BATCH,
            "num_workers": 0 if args.cpu_smoke else args.workers,
            "dataset": {
                "img_folder": img_folder,
                "ann_file": train_ann.as_posix(),
                "transforms": {
                    "ops": ops,
                    "policy": {"name": "stop_epoch", "epoch": aug_off, "ops": stop_ops},
                },
            },
            "collate_fn": {"base_size": RES, "stop_epoch": aug_off},
        },
        "val_dataloader": {
            "total_batch_size": 2 if args.cpu_smoke else BATCH,
            "num_workers": 0 if args.cpu_smoke else args.workers,
            "dataset": {
                "img_folder": img_folder,
                "ann_file": eval_ann.as_posix(),
                "transforms": {
                    "ops": [
                        {"type": "Resize", "size": [RES, RES]},
                        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
                    ]
                },
            },
        },
    }
    opt = yaml.safe_load(base_cfg.read_text())["optimizer"]
    opt["lr"] = opt["lr"] * lr_scale
    for g in opt["params"]:
        if "lr" in g:
            g["lr"] = g["lr"] * lr_scale
    cfg["optimizer"] = opt
    cfg_path = out / f"{args.run}.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    resume = out / "last.pth"
    cmd = [sys.executable, "train.py", "-c", str(cfg_path), f"--seed={args.seed}"]
    cmd += ["--device", "cpu"] if args.cpu_smoke else ["--use-amp"]
    cmd += ["-r", str(resume)] if resume.exists() else ["-t", str(weights)]
    print(args.run, "training started", counts, flush=True)
    t_train = time.time()
    with open(out / "train_stdout.txt", "a") as log:
        rc = subprocess.run(cmd, cwd=repo, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        print((out / "train_stdout.txt").read_text()[-4000:])
        raise SystemExit(f"{args.run}: training failed (exit {rc})")
    train_hours = (time.time() - t_train) / 3600
    rows = [json.loads(x) for x in (out / "log.txt").read_text().splitlines() if x.strip()]
    sh(sys.executable, "-c", EXPORT, "-c", str(cfg_path), "-r", str(resume), cwd=repo)
    onnx = out / "last.onnx"
    info = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "run": args.run,
        "fold": args.fold,
        "model": f"D-FINE {args.size.upper()}",
        "commit": DFINE_COMMIT,
        "start_weights": WEIGHTS.format(args.size),
        "start_weights_sha256": sha256(weights),
        "package_manifest_sha256": sha256(args.package / "manifest.json"),
        "train_annotations_sha256": sha256(train_ann),
        "eval_annotations": eval_ann.name,
        "eval_annotations_sha256": sha256(eval_ann),
        "recipe": args.recipe,
        "ops": ops,
        "oversample_touching": args.oversample_touching,
        "excluded_images": sorted(json.loads(args.exclude.read_text())) if args.exclude else [],
        "holdout_group": args.holdout_group,
        "counts": counts,
        "tile_px": RES,
        "epochs_planned": args.epochs,
        "epochs_done": max((r["epoch"] for r in rows), default=-1) + 1,
        "augmentation_off_from_epoch": aug_off,
        "total_batch_size": BATCH,
        "authors_total_batch_size": AUTHORS_BATCH[args.size],
        "lr_scale": lr_scale,
        "seed": args.seed,
        "workers": args.workers,
        "checkpoint": "last.pth (no selection)",
        "onnx_sha256": sha256(onnx),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu_smoke": args.cpu_smoke,
        "python": platform.python_version(),
        "train_hours": round(train_hours, 3),
        "run_hours": round((time.time() - t0) / 3600, 3),
    }
    (out / "run.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
