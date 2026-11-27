"""Every result number in the README traces to results/readme_numbers.json (brief section 13)."""

from __future__ import annotations

import importlib.util

from herdsight.utils.config import REPO_ROOT

spec = importlib.util.spec_from_file_location(
    "check_readme_numbers", REPO_ROOT / "scripts" / "check_readme_numbers.py"
)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def test_readme_numbers_trace_to_results() -> None:
    assert check.main([str(REPO_ROOT / "README.md")]) == 0
