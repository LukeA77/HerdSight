# Adapted from drone-ranger (https://github.com/LukeA77/drone-ranger),
# commit b4f4e4f, tests/test_config.py.
import pytest

from herdsight.utils.config import (
    CONFIG_DIR,
    ConfigFormatError,
    MissingParameterError,
    load_config,
    parse_config,
)

CONFIG_NAMES = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_every_shipped_config_parses(name):
    cfg = load_config(name)
    assert cfg.params


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_resolved_values_record_a_source(name):
    """Any value that is set must say where it came from."""
    cfg = load_config(name)
    for key, p in cfg.params.items():
        if p.resolved:
            assert p.source, f"{name}.{key} has a value but no source"


def test_require_raises_on_null_with_placeholder_name():
    cfg = parse_config("x", {"k": {"value": None, "placeholder": "FOO", "source": None}})
    with pytest.raises(MissingParameterError, match=r"\{\{FOO\}\}"):
        cfg.require("k")


def test_require_returns_resolved_value():
    cfg = parse_config("x", {"k": {"value": 3, "placeholder": "FOO", "source": "test"}})
    assert cfg.require("k") == 3
    assert cfg.unresolved() == []


def test_unknown_key_raises():
    cfg = parse_config("x", {})
    with pytest.raises(KeyError):
        cfg.get("nope")


def test_malformed_entry_rejected():
    with pytest.raises(ConfigFormatError):
        parse_config("x", {"k": 5})


def test_config_hash_changes_with_values():
    a = parse_config("x", {"k": {"value": 1, "placeholder": None, "source": "s"}})
    b = parse_config("x", {"k": {"value": 2, "placeholder": None, "source": "s"}})
    assert a.hash() != b.hash()
