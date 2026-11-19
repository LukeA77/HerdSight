import pytest

from herdsight.utils import freeze
from herdsight.utils.run_once import TestRunRefused, claim_test_run, guard_test_run


def _repo(tmp_path):
    (tmp_path / "configs" / "freeze").mkdir(parents=True)
    (tmp_path / "configs" / "freeze" / "phases.yaml").write_text(
        "phases:\n  p1: [configs/a.yaml]\n", encoding="utf-8"
    )
    (tmp_path / "configs" / "a.yaml").write_text("x: 1\n", encoding="utf-8")
    return tmp_path


def test_unfrozen_phase_refused(tmp_path):
    root = _repo(tmp_path)
    with pytest.raises(TestRunRefused, match="not frozen"):
        guard_test_run("p1", root / "results" / "test" / "p1.json", root)


def test_changed_settings_refused(tmp_path):
    root = _repo(tmp_path)
    freeze.main(["--phase", "p1"], root=root)
    (root / "configs" / "a.yaml").write_text("x: 2\n", encoding="utf-8")
    with pytest.raises(TestRunRefused, match="changed"):
        guard_test_run("p1", root / "results" / "test" / "p1.json", root)


def test_runs_once_only(tmp_path):
    root = _repo(tmp_path)
    freeze.main(["--phase", "p1"], root=root)
    result = root / "results" / "test" / "p1.json"
    claim_test_run("p1", result, root)
    assert result.is_file()
    with pytest.raises(TestRunRefused, match="already run"):
        claim_test_run("p1", result, root)
