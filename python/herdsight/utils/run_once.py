"""Guard for one-time test runs (brief section 9.5).

A test split may be scored once, after its settings are frozen. `guard_test_run` is called at the
start of every script that touches a test split. It stops the run if:

* the phase's settings were never frozen;
* any frozen file has changed since freezing; or
* a result for this test already exists (the test has been run before).

The result file is created by `claim_test_run` immediately, as a marker, before any scoring, so a
crash part-way through cannot be followed by a quiet second attempt. A rerun after a genuine crash
(for example, out of memory) needs the marker removed by hand and the reason logged in
docs/decisions.md.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from herdsight.utils.config import REPO_ROOT
from herdsight.utils.freeze import changed_files


class TestRunRefused(RuntimeError):  # noqa: N818 - the name reads as the event it reports
    """The one-time test run is not allowed to proceed."""

    __test__ = False  # not a pytest test class


def guard_test_run(phase: str, result_path: Path, root: Path = REPO_ROOT) -> None:
    """Raise TestRunRefused unless the phase is frozen, unchanged, and not yet run."""
    manifest = root / "results" / "frozen" / f"{phase}.json"
    if not manifest.is_file():
        raise TestRunRefused(
            f"phase {phase!r} is not frozen: run scripts/freeze.py --phase {phase}"
        )
    changed = changed_files(phase, root)
    if changed:
        raise TestRunRefused(f"frozen settings for {phase!r} changed since freezing: {changed}")
    if result_path.exists():
        raise TestRunRefused(f"test already run: {result_path} exists; test splits run once")


def claim_test_run(phase: str, result_path: Path, root: Path = REPO_ROOT) -> None:
    """Check the guard, then write the marker so the test cannot be started again."""
    guard_test_run(phase, result_path, root)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    marker = {
        "status": "started",
        "phase": phase,
        "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with result_path.open("x", encoding="utf-8") as f:  # "x" fails if it already exists
        json.dump(marker, f, indent=1)
