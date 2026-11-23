"""Degradation and pass rule of the resolution study (herdsight.evaluate.resolution)."""

import numpy as np
import pytest
from PIL import Image

from herdsight.evaluate.resolution import ScaleResult, degrade, image_seed, smallest_passing


def _image(w=200, h=100):
    x = np.linspace(0, 255, w, dtype=np.uint8)
    arr = np.repeat(x[None, :, None], h, axis=0).repeat(3, axis=2)
    return Image.fromarray(arr)


def test_size_follows_scale():
    assert degrade(_image(), 0.25, 0.5, 2.0, 85, 1).size == (50, 25)


def test_same_seed_gives_the_same_image():
    a = np.asarray(degrade(_image(), 0.5, 0.5, 2.0, 85, 7))
    b = np.asarray(degrade(_image(), 0.5, 0.5, 2.0, 85, 7))
    assert np.array_equal(a, b)


def test_other_seed_changes_the_noise():
    a = np.asarray(degrade(_image(), 0.5, 0.5, 2.0, 85, 7))
    b = np.asarray(degrade(_image(), 0.5, 0.5, 2.0, 85, 8))
    assert not np.array_equal(a, b)


def test_brightness_is_kept():
    out = np.asarray(degrade(_image(), 0.5, 0.5, 2.0, 85, 3), float)
    assert out.mean() == pytest.approx(np.asarray(_image(), float).mean(), abs=2.0)


@pytest.mark.parametrize("scale", [0.0, -0.5, 1.5])
def test_bad_scale_is_refused(scale):
    with pytest.raises(ValueError):
        degrade(_image(), scale, 0.5, 2.0, 85, 1)


def test_image_seed_depends_on_name_and_scale():
    base = image_seed("a.jpg", 0.5, 11)
    assert base == image_seed("a.jpg", 0.5, 11)
    assert base != image_seed("b.jpg", 0.5, 11)
    assert base != image_seed("a.jpg", 0.25, 11)


def test_pass_rule_uses_point_and_interval_separately():
    rows = [
        ScaleResult(1.0, -0.01, -0.02),
        ScaleResult(0.5, -0.03, -0.06),
        ScaleResult(0.25, -0.08, -0.12),
    ]
    assert smallest_passing(rows, 0.05) == (0.5, 1.0)


def test_a_pass_below_a_failure_does_not_count():
    rows = [
        ScaleResult(1.0, -0.01, -0.02),
        ScaleResult(0.5, -0.09, -0.12),
        ScaleResult(0.25, -0.01, -0.02),
    ]
    assert smallest_passing(rows, 0.05) == (1.0, 1.0)


def test_nothing_passes():
    assert smallest_passing([ScaleResult(1.0, -0.2, -0.3)], 0.05) == (None, None)
