# Adapted from drone-ranger (https://github.com/LukeA77/drone-ranger),
# commit b4f4e4f, tests/test_env_and_cli.py (environment part only).
from herdsight.utils.env_report import env_report


def test_env_report_has_required_fields():
    r = env_report()
    for key in ("cpu_model", "logical_cores", "memory_gb", "gpus", "os", "python", "packages"):
        assert key in r
    assert r["packages"]["numpy"]
