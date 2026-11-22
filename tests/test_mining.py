"""Nested hard-negative mining (scripts/mine_hard_negatives.py; D-34): selection rules."""

import sys

import numpy as np

from herdsight.detect.tiling import Tile
from herdsight.utils.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import mine_hard_negatives as mh  # noqa: E402

FOLDS = ["fold0", "fold1", "fold2"]


def test_a_tile_touching_any_animal_is_not_animal_free():
    t = Tile(0, 0, 640, 640, 0, 0, 640, 640)
    assert mh.animal_free(t, np.zeros((0, 4)))
    assert mh.animal_free(t, [[700, 700, 800, 800]])
    assert not mh.animal_free(t, [[630, 100, 700, 160]])  # only 10 px inside: still an animal


def _cand(tile, fold, model_fold, score, source=None):
    return {
        "tile": tile,
        "fold": fold,
        "model_fold": model_fold,
        "score": score,
        "source": source or tile,
    }


def test_each_fold_gets_negatives_only_from_models_that_never_saw_it():
    cands = [
        _cand(f"{f}-{m}-{i}", f, m, 0.9 - i / 100)
        for f in FOLDS
        for m in FOLDS
        if m != f
        for i in range(5)
    ]
    for k in FOLDS:
        for _, pool in mh.ranked(cands, k, FOLDS):
            assert pool and all(c["fold"] != k and c["model_fold"] != k for c in pool)
            assert all(c["fold"] != c["model_fold"] for c in pool)
            scores = [c["score"] for c in pool]
            assert scores == sorted(scores, reverse=True)


def test_take_caps_tiles_per_photo_and_skips_rejected():
    pool = [_cand(f"t{i}", "fold1", "fold0", 1 - i / 10, source="photoA") for i in range(4)]
    pool += [_cand("u0", "fold1", "fold0", 0.5, source="photoB")]
    got = [c["tile"] for c in mh.take(pool, 10)]
    assert got == ["t0", "t1", "u0"]  # at most PER_IMAGE (2) from photo A
    assert [c["tile"] for c in mh.take(pool, 10, skip={"t0"})] == ["t1", "t2", "u0"]
    assert len(mh.take(pool, 1)) == 1


def test_the_photo_choice_is_fixed_per_photo():
    names = [f"D:/x/DJI_{i:04d}_V.JPG" for i in range(400)]
    a = [mh.image_draw(n) for n in names]
    assert a == [mh.image_draw(n) for n in reversed(names)][::-1]
    assert 0.4 < np.mean(np.array(a) < 0.5) < 0.6
