"""Rules that turn SAM results into oriented training labels (herdsight.labels.sam_final)."""

from collections import Counter

import numpy as np
import pytest

from herdsight.labels.sam_final import MIN_PER_CELL, final_obb, select_for_review

BOX = [10.0, 20.0, 50.0, 40.0]


def _label(obb=(31.0, 29.0, 40.0, 15.0, 0.5), flags=()):
    return {"obb": None if obb is None else list(obb), "flags": list(flags)}


def test_good_label_keeps_sam_box():
    assert final_obb(BOX, _label(), False) == pytest.approx([31, 29, 40, 15, 0.5])


def test_centre_rule_keeps_angle_and_size_only():
    assert final_obb(BOX, _label(), True) == pytest.approx([30, 30, 40, 15, 0.5])


@pytest.mark.parametrize("label", [_label(None, ["no_mask"]), _label(flags=["escapes_box"])])
def test_flagged_label_falls_back_and_is_not_dropped(label):
    assert final_obb(BOX, label, False) == pytest.approx([30, 30, 40, 20, 0])


def test_tall_fallback_is_in_canonical_form():
    out = final_obb([0.0, 0.0, 10.0, 30.0], _label(None), False)
    assert out[2] == pytest.approx(30) and out[3] == pytest.approx(10)
    assert -np.pi / 2 <= out[4] < np.pi / 2


def _rows():
    rows = []
    for f, n_good, n_flag in (("fold0", 300, 10), ("fold1", 200, 3), ("fold2", 100, 40)):
        labels = [_label() for _ in range(n_good)] + [_label(flags=["low_fill"])] * n_flag
        rows.append({"image": f"{f}.jpg", "fold": f, "labels": labels})
    return rows


def test_review_sample_is_stratified_unique_and_repeatable():
    rows = _rows()
    a = select_for_review(rows, 100, seed=4)
    assert len(a) == 100 == len(set(a))
    assert a == select_for_review(rows, 100, seed=4)
    fold = {r["image"]: r["fold"] for r in rows}
    labels = {r["image"]: r["labels"] for r in rows}
    cells = Counter((fold[i], bool(labels[i][k]["flags"])) for i, k in a)
    assert len(cells) == 6
    assert cells[("fold1", True)] == 3  # a cell smaller than the floor is taken whole
    assert all(v >= MIN_PER_CELL for c, v in cells.items() if c != ("fold1", True))


def test_too_small_a_sample_is_refused():
    with pytest.raises(ValueError):
        select_for_review(_rows(), 10)
