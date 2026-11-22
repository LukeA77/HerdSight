import pytest

from herdsight.utils import freeze


def _repo(tmp_path, phases_yaml):
    (tmp_path / "configs" / "freeze").mkdir(parents=True)
    (tmp_path / "configs" / "freeze" / "phases.yaml").write_text(phases_yaml, encoding="utf-8")
    (tmp_path / "configs" / "a.yaml").write_text("x: 1\n", encoding="utf-8")
    return tmp_path


def test_freeze_then_check_detects_a_change(tmp_path):
    root = _repo(tmp_path, "phases:\n  p1: [configs/a.yaml]\n")
    assert freeze.main(["--phase", "p1"], root=root) == 0
    assert freeze.main(["--phase", "p1", "--check"], root=root) == 0
    (root / "configs" / "a.yaml").write_text("x: 2\n", encoding="utf-8")
    assert freeze.main(["--phase", "p1", "--check"], root=root) == 1


def test_deleted_frozen_file_counts_as_changed(tmp_path):
    root = _repo(tmp_path, "phases:\n  p1: [configs/a.yaml]\n")
    freeze.main(["--phase", "p1"], root=root)
    (root / "configs" / "a.yaml").unlink()
    assert freeze.changed_files("p1", root) == ["configs/a.yaml"]


def test_missing_file_cannot_be_frozen(tmp_path):
    root = _repo(tmp_path, "phases:\n  p1: [configs/missing.yaml]\n")
    with pytest.raises(freeze.FreezeError, match="not found"):
        freeze.main(["--phase", "p1"], root=root)


def test_unknown_phase_is_refused(tmp_path):
    root = _repo(tmp_path, "phases:\n  p1: [configs/a.yaml]\n")
    assert freeze.main(["--phase", "nope"], root=root) == 2


def test_malformed_phase_list_rejected(tmp_path):
    root = _repo(tmp_path, "phases:\n  p1: configs/a.yaml\n")
    with pytest.raises(freeze.FreezeError):
        freeze.load_phases(root)


def test_shipped_phase_file_is_valid():
    freeze.load_phases()


def test_line_endings_do_not_change_the_hash(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"x: 1\ny: 2\n")
    lf = freeze.sha256(tmp_path, "a.txt")
    (tmp_path / "a.txt").write_bytes(b"x: 1\r\ny: 2\r\n")
    assert freeze.sha256(tmp_path, "a.txt") == lf
    (tmp_path / "a.txt").write_bytes(b"x: 2\ny: 2\n")
    assert freeze.sha256(tmp_path, "a.txt") != lf


def test_binary_files_are_hashed_as_they_are(tmp_path):
    (tmp_path / "w.bin").write_bytes(b"\0\r\n")
    a = freeze.sha256(tmp_path, "w.bin")
    (tmp_path / "w.bin").write_bytes(b"\0\n")
    assert freeze.sha256(tmp_path, "w.bin") != a
