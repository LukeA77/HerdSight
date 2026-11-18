"""Parity: the C++ core (herdsight._core) must match the Python reference (herdsight.geo).

The C++ module is optional on machines without a compiler, so this test skips when it is not
built, unless HERDSIGHT_REQUIRE_CORE=1 is set. CI sets it, so there the check cannot be
skipped.
"""

import os

import numpy as np
import pytest

from herdsight.geo import frames as fr

if os.environ.get("HERDSIGHT_REQUIRE_CORE") == "1":
    from herdsight import _core
else:
    _core = pytest.importorskip("herdsight._core")

TOL = 1e-12
N = 10_000


@pytest.fixture(scope="module")
def angles():
    rng = np.random.default_rng(20261118)
    return np.column_stack(
        [
            rng.uniform(-np.pi, np.pi, N),
            rng.uniform(-1.5, 1.5, N),
            rng.uniform(-np.pi, np.pi, N),
        ]
    )


@pytest.mark.parametrize("name", ["rot_x", "rot_y", "rot_z"])
def test_axis_rotations_match(name, angles):
    py_f, cpp_f = getattr(fr, name), getattr(_core, name)
    worst = max(np.abs(cpp_f(a) - py_f(a)).max() for a in angles[:, 0])
    assert worst <= TOL


def test_R_wb_from_ypr_matches(angles):
    worst = max(np.abs(_core.R_wb_from_ypr(*a) - fr.R_wb_from_ypr(*a)).max() for a in angles)
    assert worst <= TOL


def test_ypr_from_R_wb_matches(angles):
    worst = 0.0
    for a in angles:
        R = fr.R_wb_from_ypr(*a)
        diff = np.abs(np.subtract(_core.ypr_from_R_wb(R), fr.ypr_from_R_wb(R))).max()
        worst = max(worst, diff)
    assert worst <= TOL
