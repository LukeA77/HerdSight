"""Regression: stored simulated flights still give the saved home-base results (Phase 5).

Runs only where the stored flights are in the storage folder (HERDSIGHT_STORAGE); skipped elsewhere,
for example in CI. A failure means counting, ranges, status or wording changed: if that was meant,
save new results with `python scripts/flight_replay.py update --note "<why>"`.
"""

from __future__ import annotations

import importlib.util

import pytest

from herdsight.utils.config import REPO_ROOT

spec = importlib.util.spec_from_file_location(
    "flight_replay", REPO_ROOT / "scripts" / "flight_replay.py"
)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


@pytest.mark.skipif(not replay.available(), reason="stored flights not in the storage folder")
def test_stored_flights_give_the_saved_results() -> None:
    assert replay.check() == []


def test_saved_results_cover_every_listed_flight() -> None:
    import json

    golden = json.loads(replay.GOLDEN.read_text(encoding="utf-8"))
    assert set(golden["flights"]) == set(replay.FLIGHTS)
    assert golden["last_change"]
