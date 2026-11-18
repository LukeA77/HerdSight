"""Find the storage folder for heavy files: datasets, model weights, run outputs and caches.

Heavy files never go in the repository or on the laptop's system disk. Their location comes
from the HERDSIGHT_STORAGE environment variable, so the same code runs on a laptop with an
external drive, in CI, and in a cloud notebook (where the variable points at its working disk).

If the variable is missing, or the folder does not exist (for example, the drive is not
connected), the code stops with a clear message instead of quietly writing somewhere else.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from herdsight.utils.config import REPO_ROOT

ENV_VAR = "HERDSIGHT_STORAGE"

#: Standard layout inside the storage folder.
LAYOUT = {
    "raw": Path("datasets/raw"),
    "processed": Path("datasets/processed"),
    "weights": Path("weights"),
    "runs": Path("runs"),
    "cache": Path("cache"),
    "exports": Path("exports"),
}


class StorageNotFoundError(RuntimeError):
    """The storage folder is not configured, not reachable, or in an unsafe place."""


def storage_root(env: Mapping[str, str] | None = None, repo_root: Path = REPO_ROOT) -> Path:
    """Return the storage folder named by HERDSIGHT_STORAGE.

    Raises StorageNotFoundError if the variable is unset, the folder does not exist, or the
    folder is inside the repository (where large files could be committed by mistake).
    """
    env = os.environ if env is None else env
    raw = env.get(ENV_VAR, "").strip()
    if not raw:
        raise StorageNotFoundError(
            f"{ENV_VAR} is not set. Set it to the storage folder for datasets and weights "
            f"(for example D:\\HerdSight-storage), then open a new terminal."
        )
    root = Path(raw).expanduser().resolve()
    if not root.is_dir():
        raise StorageNotFoundError(
            f"{ENV_VAR} points to {root}, which does not exist. Is the external drive connected?"
        )
    repo = repo_root.resolve()
    if root == repo or repo in root.parents:
        raise StorageNotFoundError(
            f"{ENV_VAR} points inside the repository ({root}). Large files must live outside it."
        )
    return root


def storage_path(
    kind: str,
    *parts: str,
    create: bool = False,
    env: Mapping[str, str] | None = None,
    repo_root: Path = REPO_ROOT,
) -> Path:
    """Path inside the storage folder, e.g. storage_path("raw", "icaerus").

    `kind` is one of LAYOUT's keys. With create=True the folder is made if missing.
    """
    if kind not in LAYOUT:
        raise KeyError(f"unknown storage kind {kind!r}; expected one of {sorted(LAYOUT)}")
    path = storage_root(env, repo_root) / LAYOUT[kind]
    path = path.joinpath(*parts)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
