"""Animal ovals for training tiles and coat darkening (gpu/tile_ovals.py), on synthetic cases."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "tile_ovals", Path(__file__).parents[1] / "gpu" / "tile_ovals.py"
)
tov = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tov)


def test_parse_tile_name_and_photo_key():
    key, x0, y0 = tov.parse_tile_name(
        "Derval__DJI_202308091442_012__DJI_20230809145607_0007_V__448_0.jpg"
    )
    assert (key, x0, y0) == ("Derval/DJI_202308091442_012/DJI_20230809145607_0007_V", 448, 0)
    assert tov.photo_key("Derval/DJI_202308091442_012/DJI_20230809145607_0007_V.JPG") == key
    assert tov.parse_tile_name("hardneg/Mauron__f__s__0_896.jpg")[1:] == (0, 896)
    with pytest.raises(ValueError):
        tov.parse_tile_name("not_a_tile.jpg")


def test_tile_ovals_moves_into_tile_and_keeps_only_centres_inside():
    obbs = {"F/f/p": [[500.0, 100.0, 80.0, 40.0, 0.3], [1200.0, 100.0, 80.0, 40.0, 0.0]]}
    out = tov.tile_ovals(
        obbs, ["F__f__p__448_0.jpg", "F__f__p__0_640.jpg", "G__g__q__0_0.jpg"], 640
    )
    assert list(out) == ["F__f__p__448_0.jpg"]  # second tile has no centre inside; G has no labels
    assert out["F__f__p__448_0.jpg"] == [[52.0, 100.0, 80.0, 40.0, 0.3]]


def test_oval_alpha_inside_outside_and_rotation():
    a = tov.oval_alpha(100, 100, [50, 50, 60, 20, 0.0])
    assert a[50, 50] == 1.0 and a[50, 75] == 1.0  # along the long axis, inside
    assert a[50, 95] == 0.0 and a[75, 50] == 0.0  # beyond the tip, and across the short axis
    r = tov.oval_alpha(100, 100, [50, 50, 60, 20, math.pi / 2])
    assert r[75, 50] == 1.0 and r[50, 75] == 0.0  # rotated 90 degrees: long axis now vertical
    area = a.sum()
    expected = math.pi * (30 * 0.95) * (10 * 0.95)
    assert abs(area - expected) / expected < 0.08


def test_darken_coats_changes_only_the_animal():
    rng = np.random.default_rng(0)
    img = rng.integers(60, 200, (80, 120, 3)).astype(np.uint8)
    oval = [60, 40, 50, 24, 0.4]
    out = tov.darken_coats(img, [oval], factor=0.4, desaturate=0.5)
    inside = tov.oval_alpha(120, 80, oval) == 1.0
    outside = tov.oval_alpha(120, 80, oval) == 0.0
    assert np.array_equal(out[outside], img[outside])
    assert out[inside].mean() < 0.5 * img[inside].mean()
    spread_in = out[inside].astype(float).std(axis=1).mean()
    assert spread_in < img[inside].astype(float).std(axis=1).mean()  # less colour inside
    with pytest.raises(ValueError):
        tov.darken_coats(img, [oval], factor=0.0, desaturate=0.5)


def test_oval_box_bounds_the_oval():
    oval = [50.0, 50.0, 60.0, 20.0, 0.7]
    x0, y0, x1, y1 = tov.oval_box(oval)
    a = tov.oval_alpha(100, 100, oval, shrink=1.0, feather_px=0.01)
    ys, xs = np.nonzero(a > 0.5)
    assert x0 - 1 <= xs.min() and xs.max() <= x1 + 1 and y0 - 1 <= ys.min() and ys.max() <= y1 + 1
    assert tov.oval_box([10, 20, 40, 10, 0.0]) == [-10.0, 15.0, 30.0, 25.0]
