"""Choosing second-look spots (herdsight.count.spots) on small made-up flights."""

import numpy as np
import pytest

from herdsight.count.spots import choose_spots

THR = 0.5


def cow(x, y, length=1.2, width=0.6):
    return [x - length / 2, y - width / 2, x + length / 2, y + width / 2]


def spread_herd(x0, n=5, spacing=4.0):
    """n animals well apart: a confident, easy group."""
    return [cow(x0 + i * spacing, 0.0) for i in range(n)]


def test_clear_spread_herd_needs_no_second_look():
    boxes = spread_herd(0.0)
    assert choose_spots(boxes, [0.9] * len(boxes), THR) == []


def test_oversized_box_makes_a_spot():
    boxes = spread_herd(0.0) + [cow(22.0, 0.0, length=2.4)]  # two animals in one box
    spots = choose_spots(boxes, [0.9] * len(boxes), THR)
    assert len(spots) == 1 and "oversized box" in spots[0].reasons
    assert spots[0].hidden_estimate == 1


def test_faint_detections_make_a_spot():
    boxes = spread_herd(0.0) + [cow(2.0, 3.0), cow(6.0, 3.0)]
    scores = [0.9] * 5 + [0.2, 0.3]
    spots = choose_spots(boxes, scores, THR)
    assert len(spots) == 1 and spots[0].reasons == ("faint detections",)
    assert spots[0].animals == 5 and spots[0].hidden_estimate == 2


def test_touching_group_makes_a_spot():
    boxes = [cow(i * 1.25, 0.0) for i in range(4)] + spread_herd(100.0)
    spots = choose_spots(boxes, [0.9] * len(boxes), THR)
    assert len(spots) == 1 and spots[0].reasons == ("touching animals",)
    assert abs(spots[0].centre[0] - 1.875) < 1e-9


def test_small_groups_are_ignored():
    boxes = [cow(0.0, 0.0), cow(1.25, 0.0)]  # touching, but only two animals
    assert choose_spots(boxes, [0.9, 0.9], THR) == []


def test_nearby_spots_are_merged_and_capped():
    tight = [cow(i * 1.25, 0.0) for i in range(3)]
    boxes = []
    for k in range(8):  # eight touching groups 30 m apart along a line
        boxes += [[b[0] + k * 30, b[1], b[2] + k * 30, b[3]] for b in tight]
    spots = choose_spots(boxes, [0.9] * len(boxes), THR, max_spots=5)
    assert len(spots) <= 5
    # 30 m apart with a 60 m footprint overlaps by 50%, so neighbours are merged
    assert sum(s.animals for s in spots) <= 24 and spots[0].animals > 3


def test_ranked_by_hidden_animals():
    a = [cow(i * 4.0, 0.0) for i in range(3)] + [cow(0.0, 3.0), cow(4.0, 3.0)]
    b = [cow(500 + i * 4.0, 0.0) for i in range(3)] + [
        cow(500.0, 3.0),
        cow(504.0, 3.0),
        cow(508.0, 3.0),
    ]
    scores = [0.9] * 3 + [0.2] * 2 + [0.9] * 3 + [0.2] * 3
    spots = choose_spots(a + b, scores, THR)
    assert [s.hidden_estimate for s in spots] == [3, 2]


def test_mismatched_scores_rejected():
    with pytest.raises(ValueError):
        choose_spots(np.zeros((2, 4)), [0.9], THR)
