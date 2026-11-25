"""Phase 2 step G10 on a free Kaggle GPU: the same work as gpu/runpod/job_g10.sh (recipe phase1).

D-FINE N, reference recipe (D-50), the three cross-validation folds, 72 epochs each, each model
exported to ONNX and checked on its monitor set. Runs as a Kaggle script with GPU and Internet on.
Inputs (private Kaggle datasets): herdsight-tiles-v1 and a code snapshot of the repository at a
fixed commit (a `git archive` zip). Outputs go to /kaggle/working/out/g10n_<fold>/, as on RunPod,
so full-image scoring (job_fullscore.sh, HS_KINDS=g10n) reads them the same way.

A fold that would not finish inside the session's time budget is not started; run again with
HS_FOLDS set to the folds still missing.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/tmp") if Path("/kaggle/tmp").exists() else Path("/tmp/hs")
OUT = Path("/kaggle/working/out")
FOLDS = os.environ.get("HS_FOLDS", "fold0,fold1,fold2").split(",")
MAX_HOURS = 11.3
T0 = time.time()


def sh(*cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


def find_zip(prefix):
    for p in INPUT.rglob("*.zip"):
        if p.name.startswith(prefix):
            return p
    return None


def code_root():
    dest = WORK / "HerdSight"
    for p in INPUT.rglob("train_dfine.py"):  # Kaggle may have unzipped the snapshot
        if p.parent.name == "gpu" and (p.parent / "eval_tiles.py").exists():
            return p.parent.parent
    if not dest.exists():
        z = find_zip("code_")
        if z is None:
            raise SystemExit("code snapshot not found under /kaggle/input")
        zipfile.ZipFile(z).extractall(dest)
    return dest


def package(name):
    for d in INPUT.rglob("manifest.json"):
        if (d.parent / "images").is_dir() and json.load(open(d)).get("name") == name:
            return d.parent
    z = find_zip(name)
    if z is None:
        raise SystemExit(f"package {name} not found under /kaggle/input")
    dest = WORK / "packages"
    if not (dest / name).exists():
        zipfile.ZipFile(z).extractall(dest)
    return dest / name


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    repo = code_root()
    pkg = package("herdsight_tiles_v1")
    try:
        sh(sys.executable, "-m", "pip", "install", "-q", "onnxruntime-gpu==1.22.0")
    except subprocess.CalledProcessError:
        sh(sys.executable, "-m", "pip", "install", "-q", "onnxruntime")
    workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"tiles: {pkg}, workers: {workers}, folds: {FOLDS}", flush=True)
    fold_hours, done = [], []
    for fold in FOLDS:
        used = (time.time() - T0) / 3600
        if fold_hours and used + max(fold_hours) * 1.05 > MAX_HOURS:
            print(f"{fold} not started (time budget; {used:.2f} h used)", flush=True)
            continue
        t = time.time()
        run = f"g10n_{fold}"
        sh(
            sys.executable,
            repo / "gpu" / "train_dfine.py",
            "--run",
            run,
            "--package",
            pkg,
            "--fold",
            fold,
            "--size",
            "n",
            "--epochs",
            "72",
            "--recipe",
            "phase1",
            "--workers",
            workers,
            "--work",
            WORK / "work",
            "--dfine",
            WORK / "D-FINE",
        )
        sh(
            sys.executable,
            repo / "gpu" / "eval_tiles.py",
            "--onnx",
            WORK / "work" / run / "last.onnx",
            "--ann",
            pkg / f"monitor_{fold}.json",
            "--images",
            pkg / "images",
            "--out",
            OUT / f"{run}_monitor_eval.json",
        )
        (OUT / run).mkdir(exist_ok=True)
        for f in ("run.json", "log.txt", "train_stdout.txt", f"{run}.yml", "last.onnx"):
            shutil.copy2(WORK / "work" / run / f, OUT / run / f)
        fold_hours.append((time.time() - t) / 3600)
        done.append(fold)
        print(f"{run} done in {fold_hours[-1]:.2f} h", flush=True)
    summary = {
        "job": "g10",
        "recipe": "phase1",
        "platform": "kaggle",
        "folds_done": done,
        "fold_hours": [round(h, 2) for h in fold_hours],
        "session_hours": round((time.time() - T0) / 3600, 2),
    }
    (OUT / "session.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
