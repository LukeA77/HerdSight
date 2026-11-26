"""Phase 2 step G13, farm onboarding curve, for one farm on a free Kaggle GPU.

Settings (environment, or the defaults below): HS_FARM, HS_POOL_DAYS (space-separated survey dates
whose photos may be given to us), HS_KS (photo counts, e.g. "50 200"). Inputs (private Kaggle
datasets): herdsight-tiles-v1 (full-resolution tiles, as for the home-base detector), the code
snapshot, and the leave-one-farm-out models (herdsight-rp-g6ref-3, the k = 0 point).

For each k it trains D-FINE S with the reference recipe (D-50) on every tile of the other farms plus
every tile of k photos of the farm's pool days (gpu/make_onboarding_sets.py), then scores it and
the k = 0 model on every tile of the farm's other days (gpu/eval_tiles.py). Outputs go to
/kaggle/working/out/. Public data; not a field result.
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
FARM = os.environ.get("HS_FARM", "Jalogny")
POOL = os.environ.get("HS_POOL_DAYS", "20230623 20230627").split()
KS = [int(k) for k in os.environ.get("HS_KS", "50 200").split()]
MAX_HOURS = 11.3
T0 = time.time()


def sh(*cmd: object, cwd: Path | None = None) -> None:
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, cwd=cwd)


def code_root() -> Path:
    for p in INPUT.rglob("train_dfine.py"):
        if p.parent.name == "gpu" and (p.parent / "eval_tiles.py").exists():
            dest = WORK / "HerdSight"
            if not dest.exists():
                shutil.copytree(p.parent.parent, dest)
            return dest
    z = next(INPUT.rglob("code_*.zip"))
    dest = WORK / "HerdSight"
    if not dest.exists():
        zipfile.ZipFile(z).extractall(dest)
    return dest


def package(name: str) -> Path:
    for d in INPUT.rglob("manifest.json"):
        if (d.parent / "images").is_dir() and json.load(open(d)).get("name") == name:
            return d.parent
    z = next(p for p in INPUT.rglob("*.zip") if p.name.startswith(name))
    dest = WORK / "packages"
    if not (dest / name).exists():
        zipfile.ZipFile(z).extractall(dest)
    return dest / name


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    repo = code_root()
    # the onboarding set builder ships with this script's commit, not the code snapshot
    here = Path(__file__).resolve().parent
    for f in ("make_onboarding_sets.py",):
        if (here / f).exists():
            shutil.copy2(here / f, repo / "gpu" / f)
    pkg = package("herdsight_tiles_v1")
    sets = WORK / "onb"
    sh(sys.executable, "make_onboarding_sets.py", "--package", pkg, "--out", sets, "--farm", FARM,
       "--pool-days", *POOL, "--k", *KS, cwd=repo / "gpu")  # fmt: skip
    shutil.copy2(sets / f"onb_{FARM}_photos.json", OUT)
    try:
        sh(sys.executable, "-m", "pip", "install", "-q", "onnxruntime-gpu==1.22.0")
    except subprocess.CalledProcessError:
        sh(sys.executable, "-m", "pip", "install", "-q", "onnxruntime")
    test = sets / f"test_onb_{FARM}.json"

    def evaluate(onnx: Path, name: str) -> None:
        sh(sys.executable, repo / "gpu" / "eval_tiles.py", "--onnx", onnx, "--ann", test,
           "--images", pkg / "images", "--out", OUT / f"{name}_test_eval.json")  # fmt: skip

    lofo = next(INPUT.rglob(f"g6ref_{FARM}/last.onnx"), None)
    if lofo is not None:
        evaluate(lofo, f"g13_{FARM}_k0")
    else:
        print("leave-one-farm-out model not attached: k = 0 not scored here", flush=True)
    workers = max(1, (os.cpu_count() or 2) - 1)
    hours = []
    for k in KS:
        used = (time.time() - T0) / 3600
        if hours and used + max(hours) * 1.05 > MAX_HOURS:
            print(f"k={k} not started (time budget; {used:.2f} h used)", flush=True)
            continue
        t = time.time()
        run = f"g13_{FARM}_k{k}"
        sh(sys.executable, repo / "gpu" / "train_dfine.py", "--run", run, "--package", sets,
           "--fold", f"onb_{FARM}_k{k}", "--size", "s", "--epochs", "72", "--recipe", "phase1",
           "--workers", workers, "--work", WORK / "work", "--dfine", WORK / "D-FINE")  # fmt: skip
        evaluate(WORK / "work" / run / "last.onnx", run)
        (OUT / run).mkdir(exist_ok=True)
        for f in ("run.json", "log.txt", f"{run}.yml", "last.onnx"):
            shutil.copy2(WORK / "work" / run / f, OUT / run / f)
        hours.append((time.time() - t) / 3600)
        print(f"{run} done in {hours[-1]:.2f} h", flush=True)
    summary = {"job": "g13", "farm": FARM, "pool_days": POOL, "ks": KS, "run_hours": hours,
               "session_hours": round((time.time() - T0) / 3600, 2)}  # fmt: skip
    (OUT / "session.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
