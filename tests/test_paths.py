import pytest

from herdsight.utils.paths import ENV_VAR, StorageNotFoundError, storage_path, storage_root


def test_unset_variable_is_a_clear_error(tmp_path):
    with pytest.raises(StorageNotFoundError, match=ENV_VAR):
        storage_root(env={}, repo_root=tmp_path / "repo")


def test_missing_folder_suggests_the_drive(tmp_path):
    env = {ENV_VAR: str(tmp_path / "not-there")}
    with pytest.raises(StorageNotFoundError, match="drive connected"):
        storage_root(env=env, repo_root=tmp_path / "repo")


def test_folder_inside_repository_is_refused(tmp_path):
    repo = tmp_path / "repo"
    (repo / "storage").mkdir(parents=True)
    with pytest.raises(StorageNotFoundError, match="inside the repository"):
        storage_root(env={ENV_VAR: str(repo / "storage")}, repo_root=repo)
    with pytest.raises(StorageNotFoundError, match="inside the repository"):
        storage_root(env={ENV_VAR: str(repo)}, repo_root=repo)


def test_storage_path_uses_the_layout_and_can_create(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    env = {ENV_VAR: str(store)}
    p = storage_path("raw", "icaerus", create=True, env=env, repo_root=tmp_path / "repo")
    assert p == store.resolve() / "datasets" / "raw" / "icaerus"
    assert p.is_dir()


def test_unknown_kind_rejected(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    with pytest.raises(KeyError):
        storage_path("models", env={ENV_VAR: str(store)}, repo_root=tmp_path / "repo")
