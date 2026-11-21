"""Export a D-FINE checkpoint downloaded from Kaggle to ONNX on this computer.

Usage: python scripts/export_dfine_onnx.py <run folder> [--python <exe>] [--repo <D-FINE clone>]

For a run whose ONNX export failed on Kaggle (gpu/train_dfine_baselines_kaggle.py). The run folder
holds last.pth, log.txt and <run>.yml as downloaded. The export is the Kaggle notebook's own EXPORT
step, read from that file so the two cannot drift, run with the D-FINE environment's Python
(default $HERDSIGHT_STORAGE/envs/dfine_smoke) in the D-FINE clone at the pinned commit
(default $HERDSIGHT_STORAGE/external/D-FINE). The config's paths are rewritten from Kaggle's to
this clone; the model part of the config is unchanged.

Writes last.onnx and updates run.json (onnx_sha256, how it was exported). If the session crashed
before run.json was written, a run.json is rebuilt from the config and log.txt, marked as rebuilt.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

from herdsight.utils.config import REPO_ROOT
from herdsight.utils.paths import storage_root

KAGGLE_SCRIPT = REPO_ROOT / "gpu" / "train_dfine_baselines_kaggle.py"


def kaggle_constant(name: str):  # noqa: ANN201
    """A top-level constant from the Kaggle notebook, read without running it."""
    tree = ast.parse(KAGGLE_SCRIPT.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == name:
                return ast.literal_eval(node.value)
            if isinstance(t, ast.Tuple) and isinstance(node.value, ast.Tuple):
                for el, val in zip(t.elts, node.value.elts, strict=True):
                    if isinstance(el, ast.Name) and el.id == name:
                        return ast.literal_eval(val)
    raise SystemExit(f"{name} not found in {KAGGLE_SCRIPT}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def local_config(cfg: dict, repo: Path, run_dir: Path) -> dict:
    """The Kaggle config with its D-FINE include pointed at the local clone."""
    out = dict(cfg)
    inc = []
    for p in cfg["__include__"]:
        rel = p[p.index("configs/") :]
        local = repo / rel
        if not local.exists():
            raise SystemExit(f"{local} missing: is the D-FINE clone at the pinned commit?")
        inc.append(local.as_posix())
    out["__include__"] = inc
    out["output_dir"] = run_dir.as_posix()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--python", type=Path, default=None)
    ap.add_argument("--repo", type=Path, default=None)
    args = ap.parse_args(argv)
    run_dir = args.run_dir.resolve()
    run = run_dir.name
    py = args.python or storage_root() / "envs" / "dfine_smoke" / "Scripts" / "python.exe"
    repo = args.repo or storage_root() / "external" / "D-FINE"
    commit = kaggle_constant("DFINE_COMMIT")
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if head != commit:
        raise SystemExit(f"D-FINE clone is at {head}, not the pinned {commit}")
    pth, cfg_file = run_dir / "last.pth", run_dir / f"{run}.yml"
    for p in (pth, cfg_file):
        if not p.exists():
            raise SystemExit(f"{p} missing")

    cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    local_cfg = run_dir / f"{run}.local_export.yml"
    local_cfg.write_text(
        yaml.safe_dump(local_config(cfg, repo, run_dir), sort_keys=False), encoding="utf-8"
    )
    onnx = run_dir / "last.onnx"
    if onnx.exists():
        onnx.unlink()
    export = kaggle_constant("EXPORT")
    print(f"exporting {run} with {py}", flush=True)
    subprocess.run(
        [str(py), "-c", export, "-c", str(local_cfg), "-r", str(pth)], cwd=repo, check=True
    )
    if not onnx.exists():
        raise SystemExit("export finished but last.onnx was not written")
    torch_version = subprocess.run(
        [str(py), "-c", "import torch; print(torch.__version__)"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    info_path = run_dir / "run.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
    else:
        rows = [json.loads(x) for x in (run_dir / "log.txt").read_text().splitlines() if x.strip()]
        epochs_done = max((r["epoch"] for r in rows), default=-1) + 1
        planned = int(cfg["epochs"])
        job, fold = run.rsplit("_", 1)
        info = {
            "run": run,
            "job": job,
            "fold": fold,
            "commit": commit,
            "smoke": 0 if planned == kaggle_constant("EPOCHS") else 1,
            "epochs_planned": planned,
            "epochs_done": epochs_done,
            "stopped_by_time_limit": epochs_done < planned,
            "checkpoint": "last.pth (no selection)",
            "run_json": "rebuilt locally from the config and log.txt: the Kaggle session stopped "
            "before writing run.json (its ONNX export ran out of memory)",
        }
    info.update(
        {
            "onnx_sha256": sha256(onnx),
            "export_error": None,
            "onnx_exported": f"locally, {datetime.now(UTC).isoformat(timespec='seconds')}, "
            f"torch {torch_version}, by scripts/export_dfine_onnx.py",
            "last_pth_sha256": sha256(pth),
        }
    )
    info_path.write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
