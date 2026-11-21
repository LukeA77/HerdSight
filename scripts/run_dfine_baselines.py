"""Run and score every Phase 1 D-FINE baseline once the Kaggle models are downloaded.

Usage: python scripts/run_dfine_baselines.py [--models <folder>] [--check-only]

Expects the six Kaggle outputs (gpu/train_dfine_baselines_kaggle.py) unzipped anywhere under
--models (default $HERDSIGHT_STORAGE/runs/baselines/models): folders named
dfine_s_baseline_fold<k> and dfine_n_onboard_fold<k>, each holding last.onnx and run.json.
Nested folders from Windows "Extract All" are fine.

Steps (each resumes where it stopped):
  1. check every model: not a smoke run, ONNX hash matches run.json, epochs done reported;
  2. home-base detector (S1, S2): scripts/run_baseline.py --detector dfine, per fold;
  3. onboard herd detector (S7): scripts/run_baseline.py --detector dfine --onboard, per fold;
  4. scoring: score_baseline.py --detector dfine, score_herds.py, score_s3.py --detector dfine,
     then judge_hypotheses.py and baseline_summary.py.
Model fold k never saw fold k's farm-date groups, so every prediction is out-of-fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from herdsight.utils.config import REPO_ROOT
from herdsight.utils.paths import storage_path

JOBS = {"dfine_s_baseline": [], "dfine_n_onboard": ["--onboard"]}
FOLDS = ("fold0", "fold1", "fold2")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_models(root: Path) -> tuple[dict[str, Path], list[str]]:
    found, problems = {}, []
    for job in JOBS:
        for fold in FOLDS:
            run = f"{job}_{fold}"
            hits = sorted(p.parent for p in root.rglob("run.json") if p.parent.name == run)
            if not hits:
                problems.append(f"{run}: not found under {root}")
                continue
            d = hits[0]
            info = json.loads((d / "run.json").read_text())
            onnx = d / "last.onnx"
            if info.get("smoke"):
                problems.append(f"{run}: is a smoke-test model, not a real run")
            elif not onnx.exists() or not info.get("onnx_sha256"):
                problems.append(
                    f"{run}: no last.onnx (Kaggle export failed?): "
                    f"python scripts/export_dfine_onnx.py {d}"
                )
            elif sha256(onnx) != info["onnx_sha256"]:
                problems.append(f"{run}: last.onnx does not match run.json (corrupt download?)")
            else:
                found[run] = onnx
                print(
                    f"{run}: OK, {info['epochs_done']}/{info['epochs_planned']} epochs, "
                    f"stopped by time limit: {info['stopped_by_time_limit']}",
                    flush=True,
                )
    return found, problems


def step(*args: str) -> None:
    cmd = [sys.executable, *args]
    print(">>", " ".join(args), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=None)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args(argv)
    root = args.models or storage_path("runs", "baselines", "models")
    found, problems = find_models(root)
    for p in problems:
        print("PROBLEM:", p, flush=True)
    if problems or args.check_only:
        return 1 if problems else 0
    for job, extra in JOBS.items():
        for fold in FOLDS:
            onnx = str(found[f"{job}_{fold}"])
            step(
                "scripts/run_baseline.py",
                "--detector",
                "dfine",
                *extra,
                "--fold",
                fold,
                "--onnx",
                onnx,
            )
    step("scripts/score_baseline.py", "--detector", "dfine")
    step("scripts/score_herds.py", "--detector", "dfine_onboard")
    step("scripts/score_s3.py", "--detector", "dfine")
    step("scripts/judge_hypotheses.py")
    step("scripts/baseline_summary.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
