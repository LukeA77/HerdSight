# Phase 1 baselines: fine-tune D-FINE once per cross-validation fold (D-26), for two jobs:
#   home-base  D-FINE S on full-resolution 640 px tiles (S1, S2; D-22)
#   onboard    D-FINE N on 640 px tiles of images reduced to a 1280 px long side (S7; D-29)
# The model for fold k is trained on the other folds only. Every model is exported to ONNX for
# local, CPU inference on its own fold's images.
#
# Runs on Kaggle (GPU on, Internet on): paste into ONE notebook cell, then Save Version ->
# Save & Run All. Inputs: PRIVATE Kaggle datasets holding herdsight_tiles_v1.zip and
# herdsight_onboard_tiles_v1.zip (made by scripts/make_training_tiles.py). Kaggle may unzip them;
# both forms are found.
# Output: /kaggle/working/<run>_<fold>.zip per run, each with last.pth, last.onnx, log.txt,
#   run.json. Put them in $HERDSIGHT_STORAGE/runs/baselines/models/ and unzip.
# Runs go one after another (home-base first). A run that would not finish in the session's time
# budget is not started; run the notebook again with RUNS cut down to the ones still missing.
#
# Recipe (authors' defaults, not tuned): the authors' custom-dataset config for each size at the
# pinned commit, started from the authors' COCO weights; tiles are not resized (Resize to 640 is a
# no-op on 640 px tiles); total batch 16 with learning rates scaled linearly from the authors'
# batch; 72 epochs, augmentation off for the last 6 (the authors' 220/20 shortened to fit a free
# session); seed 0; the LAST checkpoint is kept (no checkpoint selection on any split). The
# monitor set is logged only.
#
# Local smoke test (CPU, 2 epochs, a few tiles per job, fold0 only; checks config and export):
#   SMOKE=8 DFINE_REPO=<D-FINE checkout at the pinned commit> PACKAGES=<folder holding the two
#   unpacked packages> python gpu/train_dfine_baselines_kaggle.py
#
# Structure adapted from drone-ranger's gpu/train_dfine_hituav_kaggle.py (same author).

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

DFINE_COMMIT = "956d1709314c2c6a4df6f34de232054578a7449f"
WEIGHTS = "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_{}_coco.pth"
# job: (D-FINE size, package name, authors' total batch for that size's custom config)
JOBS = {
    "dfine_s_baseline": ("s", "herdsight_tiles_v1", 64),
    "dfine_n_onboard": ("n", "herdsight_onboard_tiles_v1", 128),
}
# (job, fold) in the order they run; cut this list down to rerun only what is missing
RUNS = [(j, f) for j in JOBS for f in ("fold0", "fold1", "fold2")]
# runs already trained in an earlier session (a full run takes about 3.2 hours on Kaggle's GPU, so
# the six do not fit in one 12-hour session)
DONE = {"dfine_s_baseline_fold0", "dfine_s_baseline_fold1", "dfine_s_baseline_fold2"}
RUNS = [(j, f) for j, f in RUNS if f"{j}_{f}" not in DONE]
RES, BATCH = 640, 16
EPOCHS, AUG_OFF = 72, 66
NUM_CLASSES = 2  # category id 1 = cattle; slot 0 unused (remap_mscoco_category False)
SEED = 0
MAX_HOURS = 11.0

SMOKE = int(os.environ.get("SMOKE", "0"))
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
ROOTS = [Path(os.environ["PACKAGES"])] if SMOKE else [Path("/kaggle/input"), WORK]
# Overrides for running outside Kaggle (gpu/runpod/); unset, they change nothing.
#   HS_WORK      working folder;  HS_PACKAGES  folder holding the unpacked packages
#   HS_RUNS      comma-separated runs to do (e.g. dfine_s_baseline_fold0), ignoring DONE
#   HS_EPOCHS    epochs (a speed benchmark, not a model to keep); augmentation stops 1 epoch early
if os.environ.get("HS_WORK"):
    WORK = Path(os.environ["HS_WORK"])
    WORK.mkdir(parents=True, exist_ok=True)
if os.environ.get("HS_PACKAGES"):
    ROOTS = [Path(os.environ["HS_PACKAGES"]), WORK]
if os.environ.get("HS_RUNS"):
    RUNS = [tuple(r.rsplit("_", 1)) for r in os.environ["HS_RUNS"].split(",")]
if os.environ.get("HS_EPOCHS"):
    EPOCHS = int(os.environ["HS_EPOCHS"])
    AUG_OFF = max(1, EPOCHS - 1)
if SMOKE:  # SMOKE_JOBS=dfine_s_baseline limits the smoke test to some jobs
    RUNS = [(j, "fold0") for j in os.environ.get("SMOKE_JOBS", ",".join(JOBS)).split(",")]
    EPOCHS, AUG_OFF = 2, 1
T0 = time.time()


def sh(*cmd, **kw):
    subprocess.run(list(cmd), check=True, **kw)


# D-FINE's own exporter, run with PyTorch's TorchScript-based ONNX exporter (dynamo=False), which
# it was written for: from PyTorch 2.9 the default exporter needs onnxscript and traces the model
# differently. Two changes that do not alter the exported model: the dummy input is 2 images, not
# 32, and gradients are off. With 32 images and gradients on, the export ran out of memory on
# Kaggle after the first full training run. (With 1 image, tracing fixes the batch size at 1.)
# scripts/export_dfine_onnx.py reads this same text to export a downloaded checkpoint locally.
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


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def find_package(name: str) -> Path:
    """The unpacked package folder whose manifest.json names `name`."""
    for r in ROOTS:
        if not r.exists():
            continue
        for d, dirs, files in os.walk(r, followlinks=True):
            if "manifest.json" in files and "images" in dirs:
                try:
                    if json.loads((Path(d) / "manifest.json").read_text())["name"] == name:
                        return Path(d)
                except (json.JSONDecodeError, KeyError):
                    continue
    for r in ROOTS:
        for d, _, files in os.walk(r, followlinks=True) if r.exists() else []:
            if f"{name}.zip" in files:
                z = Path(d) / f"{name}.zip"
                print("unpacking", z, flush=True)
                with zipfile.ZipFile(z) as f:
                    f.extractall(WORK)
                return WORK / name
    raise SystemExit(f"{name} not found: attach the Kaggle dataset holding {name}.zip")


repo = Path(os.environ["DFINE_REPO"]) if SMOKE else WORK / "D-FINE"
if not repo.exists():
    sh("git", "clone", "-q", "https://github.com/Peterande/D-FINE.git", str(repo))
sh("git", "-C", str(repo), "checkout", "-q", DFINE_COMMIT)
if not SMOKE:
    sh(sys.executable, "-m", "pip", "install", "-q", "-r", str(repo / "requirements.txt"))
    sh(sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxsim", "pycocotools", "matplotlib")
    # D-FINE asks only for >=1.6.6; 1.8.1 (released 2026-11-20) breaks its per-epoch evaluation
    # ('COCO' object has no attribute 'getAnns'). The earlier Kaggle runs and the smoke test used
    # 1.8.0.
    sh(sys.executable, "-m", "pip", "install", "-q", "faster-coco-eval==1.8.0")

import torch
import yaml

if not SMOKE and not torch.cuda.is_available():
    raise SystemExit("NO GPU FOUND. Session options -> Accelerator -> GPU, then run again.")

packages = {
    job: find_package(pkg) for job, (_, pkg, _) in JOBS.items() if any(j == job for j, _ in RUNS)
}
for job, p in packages.items():
    print(job, "package:", json.loads((p / "manifest.json").read_text()).get("counts"), flush=True)
weights = {}
for job, (size, _, _) in JOBS.items():
    w = WORK / f"dfine_{size}_coco.pth"
    if not w.exists():
        urllib.request.urlretrieve(WEIGHTS.format(size), w.with_suffix(".part"))
        os.replace(w.with_suffix(".part"), w)
    weights[job] = (w, sha256(w))

run_hours, done = [], []
for job, fold in RUNS:
    size, pkg, authors_batch = JOBS[job]
    data = packages[job]
    run = f"{job}_{fold}"
    used_h = (time.time() - T0) / 3600
    if run_hours and used_h + max(run_hours) > MAX_HOURS:
        print("not started (time budget):", run, flush=True)
        continue
    t_run = time.time()
    out = WORK / run
    out.mkdir(parents=True, exist_ok=True)
    base_cfg = repo / f"configs/dfine/custom/dfine_hgnetv2_{size}_custom.yml"
    lr_scale = BATCH / authors_batch
    train_ann, monitor_ann = data / f"train_{fold}.json", data / f"monitor_{fold}.json"
    if SMOKE:  # a few tiles, with and without animals
        for src in (train_ann, monitor_ann):
            c = json.loads(src.read_text())
            keep = {a["image_id"] for a in c["annotations"]}
            ids = (
                sorted(keep)[: SMOKE // 2]
                + [i["id"] for i in c["images"] if i["id"] not in keep][: SMOKE // 2]
            )
            c["images"] = [i for i in c["images"] if i["id"] in set(ids)]
            c["annotations"] = [a for a in c["annotations"] if a["image_id"] in set(ids)]
            dst = out / src.name
            dst.write_text(json.dumps(c))
        train_ann, monitor_ann = out / train_ann.name, out / monitor_ann.name
    cfg = {
        "__include__": [base_cfg.as_posix()],
        "output_dir": out.as_posix(),
        "num_classes": NUM_CLASSES,
        "remap_mscoco_category": False,
        "eval_spatial_size": [RES, RES],
        "epochs": EPOCHS,
        "train_dataloader": {
            "total_batch_size": 2 if SMOKE else BATCH,
            "num_workers": 0 if SMOKE else 4,
            "dataset": {
                "img_folder": (data / "images").as_posix(),
                "ann_file": train_ann.as_posix(),
                "transforms": {
                    "ops": [
                        {"type": "RandomPhotometricDistort", "p": 0.5},
                        {"type": "RandomZoomOut", "fill": 0},
                        {"type": "RandomIoUCrop", "p": 0.8},
                        {"type": "SanitizeBoundingBoxes", "min_size": 1},
                        {"type": "RandomHorizontalFlip"},
                        {"type": "Resize", "size": [RES, RES]},
                        {"type": "SanitizeBoundingBoxes", "min_size": 1},
                        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
                        {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
                    ],
                    "policy": {
                        "name": "stop_epoch",
                        "epoch": AUG_OFF,
                        "ops": ["RandomPhotometricDistort", "RandomZoomOut", "RandomIoUCrop"],
                    },
                },
            },
            "collate_fn": {"base_size": RES, "stop_epoch": AUG_OFF},
        },
        "val_dataloader": {
            "total_batch_size": 2 if SMOKE else BATCH,
            "num_workers": 0 if SMOKE else 4,
            "dataset": {
                "img_folder": (data / "images").as_posix(),
                "ann_file": monitor_ann.as_posix(),
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
    cfg_path = out / f"{run}.yml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    resume = out / "last.pth"
    cmd = [sys.executable, "train.py", "-c", str(cfg_path), f"--seed={SEED}"]
    cmd += ["--device", "cpu"] if SMOKE else ["--use-amp"]
    cmd += ["-r", str(resume)] if resume.exists() else ["-t", str(weights[job][0])]
    limit_s = int(max(0.5, MAX_HOURS - (time.time() - T0) / 3600) * 3600)
    stopped_by_time = False
    print(run, "training started", flush=True)
    with open(out / "train_stdout.txt", "a") as log:
        proc = subprocess.Popen(cmd, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
        try:
            rc = proc.wait(timeout=limit_s)
        except subprocess.TimeoutExpired:
            proc.terminate()
            rc = proc.wait()
            stopped_by_time = True
    if rc != 0 and not stopped_by_time:
        print((out / "train_stdout.txt").read_text()[-4000:])
        raise SystemExit(f"{run}: training failed (exit {rc})")
    epochs_done = 0
    if (out / "log.txt").exists():
        rows = [json.loads(x) for x in (out / "log.txt").read_text().splitlines() if x.strip()]
        epochs_done = max((r["epoch"] for r in rows), default=-1) + 1
    # an export failure must not lose the trained checkpoint: it is zipped anyway and can be
    # exported locally with scripts/export_dfine_onnx.py
    export_error = None
    try:
        sh(sys.executable, "-c", EXPORT, "-c", str(cfg_path), "-r", str(out / "last.pth"), cwd=repo)
    except subprocess.CalledProcessError as e:
        export_error = f"ONNX export failed ({e.returncode}); export last.pth locally"
        print(run, export_error, flush=True)
    onnx = out / "last.onnx"
    if export_error and onnx.exists():
        onnx.unlink()  # may be half-written
    info = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "run": run,
        "job": job,
        "fold": fold,
        "model": f"D-FINE {size.upper()}",
        "commit": DFINE_COMMIT,
        "start_weights": WEIGHTS.format(size),
        "start_weights_sha256": weights[job][1],
        "package": pkg,
        "package_manifest_sha256": sha256(data / "manifest.json"),
        "train_annotations_sha256": sha256(train_ann),
        "smoke": SMOKE,
        "tile_px": RES,
        "epochs_planned": EPOCHS,
        "epochs_done": epochs_done,
        "augmentation_off_from_epoch": AUG_OFF,
        "total_batch_size": BATCH,
        "authors_total_batch_size": authors_batch,
        "lr_scale": lr_scale,
        "seed": SEED,
        "checkpoint": "last.pth (no selection)",
        "stopped_by_time_limit": stopped_by_time,
        "onnx_sha256": sha256(onnx) if onnx.exists() and export_error is None else None,
        "export_error": export_error,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "run_hours": round((time.time() - t_run) / 3600, 2),
        "session_hours": round((time.time() - T0) / 3600, 2),
    }
    (out / "run.json").write_text(json.dumps(info, indent=1) + "\n")
    keep = WORK / "download" / run
    keep.mkdir(parents=True, exist_ok=True)
    for name in ("last.pth", "last.onnx", "log.txt", "train_stdout.txt", f"{run}.yml", "run.json"):
        if (out / name).exists():
            shutil.copy2(out / name, keep / name)
    archive = shutil.make_archive(str(WORK / run), "zip", WORK / "download", run)
    print(json.dumps(info, indent=1))
    print("run done. Download", archive, flush=True)
    run_hours.append((time.time() - t_run) / 3600)
    done.append(run)
print("RUNS DONE:", done)
print("NOT RUN:", [f"{j}_{f}" for j, f in RUNS if f"{j}_{f}" not in done])
