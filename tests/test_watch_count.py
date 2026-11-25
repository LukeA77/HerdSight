"""Counting a watched herd spot (herdsight.count.watch), on synthetic herds with known answers."""

import numpy as np
import pytest

from herdsight.count.watch import track_boxes, track_positions, watch_count

DT = 0.5
N = 90  # 45 s at 2 frames per second


def herd(n, seed=0, spacing=3.0, drift=0.02):
    """n animals on a loose grid, each wandering slowly (grazing)."""
    rng = np.random.default_rng(seed)
    start = np.array([[i % 4 * spacing, i // 4 * spacing] for i in range(n)], float)
    steps = rng.normal(0, drift, (N, n, 2))
    return start + np.cumsum(steps, axis=0)


def test_steady_herd_counts_exactly():
    pos = herd(12)
    out = watch_count(track_positions(list(pos), DT), N)
    assert (out["count"], out["low"], out["high"]) == (12, 12, 12)


def test_two_touching_animals_seen_as_one_until_they_separate():
    pos = herd(6, seed=1)
    frames = []
    for f in range(N):
        p = pos[f].copy()
        if f < 50:  # animals 0 and 1 merged into one detection at their midpoint
            p = np.vstack([(p[0] + p[1]) / 2, p[2:]])
        frames.append(p)
    pos[:, 1] = pos[:, 0] + [1.2, 0.0]  # they stand 1.2 m apart once separated
    frames[50:] = list(pos[50:])
    out = watch_count(track_positions(frames, DT), N)
    assert out["count"] == 6  # the watch finds the animal the first frames could not
    assert out["low"] == 6


def test_flickering_false_detection_is_ignored():
    pos = herd(5, seed=2)
    frames = [p.copy() for p in pos]
    for f in (10, 11, 40):  # a rock detected in three frames only
        frames[f] = np.vstack([frames[f], [20.0, 20.0]])
    out = watch_count(track_positions(frames, DT), N)
    # frames 10-11 and 40 are 14.5 s apart, beyond the 10 s a track may be unseen: two short
    # tracks, both too short to count
    assert out["count"] == 5 and out["tracks_total"] == 7


def test_animal_hidden_for_a_while_keeps_one_identity():
    pos = herd(4, seed=3)
    frames = [np.delete(p, 2, axis=0) if 30 <= f < 40 else p for f, p in enumerate(pos)]
    out = watch_count(track_positions(frames, DT, max_gap_s=10.0), N)
    assert out["count"] == 4 and out["switches_merged"] == 0


def test_long_gap_breaks_track_but_switch_merge_counts_it_once():
    pos = herd(4, seed=4)
    frames = [np.delete(p, 2, axis=0) if 20 <= f < 50 else p for f, p in enumerate(pos)]
    tracks = track_positions(frames, DT, max_gap_s=5.0)  # 15 s gap: the track is dropped
    out = watch_count(tracks, N)
    assert out["high"] == 5 and out["count"] == 4 and out["switches_merged"] == 1
    assert out["low"] == 4


def test_two_animals_that_coexist_are_never_merged():
    a = [[0.0, 0.0]] * N
    b = [[1.0, 0.0]] * N  # within the merge distance, but always together
    frames = [np.array([a[f], b[f]]) for f in range(N)]
    out = watch_count(track_positions(frames, DT, gate_m=0.4), N)
    assert out["count"] == 2


def test_bad_settings_rejected():
    with pytest.raises(ValueError):
        track_positions([np.zeros((1, 2))], dt_s=0)


def boxes_of(pos, length=1.2, width=0.6):
    """Animal boxes (metres) around centre positions."""
    p = np.asarray(pos, float).reshape(-1, 2)
    half = np.array([length / 2, width / 2])
    return np.hstack([p - half, p + half])


def test_box_tracker_counts_a_packed_herd_exactly():
    pos = herd(12, spacing=1.3)  # bodies 1.2 m long, 1.3 m apart: tightly packed
    tracks, dropped = track_boxes([boxes_of(p) for p in pos], DT)
    out = watch_count(tracks, N)
    assert (out["count"], dropped) == (12, 0)


def test_box_tracker_drops_brief_false_and_merged_detections():
    pos = herd(5, seed=5)
    frames = [boxes_of(p) for p in pos]
    frames[30] = np.vstack([frames[30], boxes_of([[30.0, 30.0]])])  # a one-frame false detection
    a, b = pos[60, 0], pos[60, 1]
    frames[60] = np.vstack(
        [boxes_of(pos[60, 2:]), [np.r_[np.minimum(a, b) - 0.6, np.maximum(a, b) + 0.6]]]
    )
    tracks, dropped = track_boxes(frames, DT)
    out = watch_count(tracks, N)
    assert out["count"] == 5 and dropped >= 1


def test_box_tracker_keeps_identity_through_a_short_hide():
    pos = herd(4, seed=6)
    frames = [boxes_of(np.delete(p, 1, axis=0) if 40 <= f < 50 else p) for f, p in enumerate(pos)]
    tracks, _ = track_boxes(frames, DT)
    out = watch_count(tracks, N)
    assert out["count"] == 4 and out["switches_merged"] == 0


def test_box_tracker_rejects_bad_settings():
    with pytest.raises(ValueError):
        track_boxes([np.zeros((0, 4))], DT, iou_min=0.0)
