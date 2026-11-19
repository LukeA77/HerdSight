import numpy as np
import pytest

from herdsight.detect.tiling import boxes_in_tile, keep_owned, tile_grid, tile_starts


def test_overlap_is_never_less_than_requested():
    for length in (700, 1120, 1200, 3956, 5280):
        s = tile_starts(length, 640, 160)
        assert all(a + 640 - b >= 160 for a, b in zip(s, s[1:], strict=False))


def test_tile_starts_cover_the_axis_with_last_tile_flush():
    s = tile_starts(5280, 640, 160)
    assert s[0] == 0 and s[-1] == 5280 - 640
    assert all(b - a <= 640 - 160 for a, b in zip(s, s[1:], strict=False))
    assert tile_starts(500, 640, 160) == [0]
    with pytest.raises(ValueError):
        tile_starts(1000, 640, 640)


def test_every_pixel_is_owned_by_exactly_one_tile():
    tiles = tile_grid(1500, 1000, 640, 160)
    xs, ys = np.meshgrid(np.arange(0, 1500, 7.3), np.arange(0, 1000, 5.1))
    owners = sum(t.owns(xs.ravel(), ys.ravel()).astype(int) for t in tiles)
    assert np.all(owners == 1)


def test_owned_region_lies_inside_the_tile():
    for t in tile_grid(1500, 1000, 640, 160):
        assert max(t.ox0, t.x0) >= t.x0 and min(t.ox1, t.x1) <= t.x1
        assert max(t.oy0, t.y0) >= t.y0 and min(t.oy1, t.y1) <= t.y1


def test_animal_in_overlap_is_counted_once_and_close_neighbours_both_survive():
    tiles = tile_grid(1120, 640, 640, 160)  # two tiles side by side, overlap x in [480, 640)
    assert len(tiles) == 2
    # Two touching animals inside the overlap (either side of its midpoint, 560), seen by both.
    animals = np.array([[530, 100, 559, 130], [561, 100, 590, 130]], float)
    kept = sum(keep_owned(t, animals).astype(int) for t in tiles)
    np.testing.assert_array_equal(kept, [1, 1])


def test_boxes_in_tile_clip_and_drop_mostly_outside():
    t = tile_grid(1120, 640, 640, 160)[1]  # x from 480 to 1120
    boxes = [[520, 10, 560, 30], [460, 10, 500, 30], [420, 10, 490, 30]]
    out, keep = boxes_in_tile(t, boxes, min_visible=0.5)
    np.testing.assert_array_equal(keep, [True, True, False])
    np.testing.assert_allclose(out, [[40, 10, 80, 30], [0, 10, 20, 30]])
