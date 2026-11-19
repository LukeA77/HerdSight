"""Freeze the settings of a phase before its one-time test run, and check the freeze.

Command line: python scripts/freeze.py --phase phase1            (write the manifest)
       python scripts/freeze.py --phase phase1 --check    (exit 1 if anything changed)

The files each phase's test run depends on are listed in configs/freeze/phases.yaml. The
manifest (results/frozen/<phase>.json) records the SHA-256 of each of those files, the git
commit, and any uncommitted changes to them. `--check` recomputes the hashes; test runs call
it first and stop if any file differs, so settings cannot drift between freezing and testing.

Based on drone-ranger's scripts/freeze.py (commit b4f4e4f); the phase lists moved from code
into a config file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from herdsight.utils.config import REPO_ROOT

PHASES_FILE = Path("configs/freeze/phases.yaml")


class FreezeError(RuntimeError):
    """The phase list is malformed or names a file that does not exist."""


def load_phases(root: Path = REPO_ROOT) -> dict[str, list[str]]:
    raw = yaml.safe_load((root / PHASES_FILE).read_text(encoding="utf-8")) or {}
    phases = raw.get("phases") or {}
    if not isinstance(phases, dict) or not all(
        isinstance(v, list) and all(isinstance(f, str) for f in v) for v in phases.values()
    ):
        raise FreezeError(f"{PHASES_FILE}: 'phases' must map each phase to a list of files")
    return phases


def sha256(root: Path, rel: str) -> str:
    path = root / rel
    if not path.is_file():
        raise FreezeError(f"frozen file not found: {rel}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(root: Path, *args: str) -> str:
    r = subprocess.run(["git", *args], capture_output=True, text=True, cwd=root)
    return r.stdout.strip()


def write_manifest(phase: str, files: list[str], root: Path = REPO_ROOT) -> dict[str, Any]:
    manifest = {
        "frozen_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "phase": phase,
        "git_commit": git(root, "rev-parse", "--short", "HEAD"),
        "uncommitted_changes_to_frozen_files": git(
            root, "status", "--porcelain", "--", *files
        ).splitlines(),
        "files": {f: sha256(root, f) for f in files},
    }
    dest = root / "results" / "frozen" / f"{phase}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def changed_files(phase: str, root: Path = REPO_ROOT) -> list[str]:
    """Files whose hash differs from the manifest (missing files count as changed)."""
    m = json.loads((root / "results" / "frozen" / f"{phase}.json").read_text(encoding="utf-8"))
    out = []
    for f, h in m["files"].items():
        try:
            if sha256(root, f) != h:
                out.append(f)
        except FreezeError:
            out.append(f)
    return out


def main(argv: list[str] | None = None, root: Path = REPO_ROOT) -> int:
    phases = load_phases(root)
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True)
    ap.add_argument("--extra", nargs="*", default=[], help="more files (e.g. chosen weights)")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    if args.phase not in phases:
        print(f"unknown phase {args.phase!r}; defined in {PHASES_FILE}: {sorted(phases)}")
        return 2
    if args.check:
        changed = changed_files(args.phase, root)
        if changed:
            print(f"FROZEN SETTINGS CHANGED for {args.phase}: {changed}")
            return 1
        print(f"{args.phase}: frozen files unchanged")
        return 0
    m = write_manifest(args.phase, phases[args.phase] + args.extra, root)
    print(json.dumps(m, indent=1))
    return 0
