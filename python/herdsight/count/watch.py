"""Counting a watched herd spot: track animals across a hover's frames, then one count and range.

The second look (_drafts/second-look-spec.md) hovers over an uncertain spot and takes frames
for 45 s. Each frame gives animal positions in ground metres (detections projected after the
frame is aligned to the first). This module turns those frames into one count:

  track_positions   joins detections frame to frame into tracks (nearest position within a gate,
                    with each track's recent velocity predicting where it will be; a track may go
                    unseen for up to max_gap_s, e.g. while hidden behind another animal)
  watch_count       confirms tracks seen often enough, merges identity switches, and returns the
                    count with its range

Rules (spec section 4): an animal is a track seen in at least min_frames frames and in at least
min_share of the frames between its first and last sighting; two confirmed tracks that never
appear in the same frame, where one ends before the other starts and the end and start are within
merge_m, are one animal (a tracking switch); count = confirmed tracks after merging; lower end =
the most confirmed animals present in one frame; upper end = confirmed tracks before merging.
Frames are never averaged: each frame is a partial view and the watch as a whole is the evidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linear_sum_assignment


@dataclass
class Track:
    id: int
    frames: list[int] = field(default_factory=list)
    positions: list[np.ndarray] = field(default_factory=list)

    @property
    def first(self) -> int:
        return self.frames[0]

    @property
    def last(self) -> int:
        return self.frames[-1]

    def predict(self, frame: int, dt_s: float) -> np.ndarray:
        """Position expected at `frame`, from the velocity over the last few sightings."""
        p = self.positions[-1]
        if len(self.frames) < 2:
            return p
        k = min(len(self.frames), 5)
        steps = self.frames[-1] - self.frames[-k]
        if steps <= 0:
            return p
        v = (self.positions[-1] - self.positions[-k]) / (steps * dt_s)
        return p + v * (frame - self.frames[-1]) * dt_s


def track_positions(
    frames: Sequence[ArrayLike],
    dt_s: float = 0.5,
    gate_m: float = 1.0,
    max_gap_s: float = 10.0,
) -> list[Track]:
    """Tracks from per-frame ground positions (each an (n, 2) array, metres)."""
    if dt_s <= 0 or gate_m <= 0 or max_gap_s < 0:
        raise ValueError("dt_s and gate_m must be positive, max_gap_s not negative")
    max_gap = int(round(max_gap_s / dt_s))
    tracks: list[Track] = []
    live: list[Track] = []
    for f, det in enumerate(frames):
        d = np.asarray(det, float).reshape(-1, 2)
        live = [t for t in live if f - t.last <= max_gap + 1]
        used = set()
        if live and len(d):
            pred = np.array([t.predict(f, dt_s) for t in live])
            cost = np.linalg.norm(pred[:, None, :] - d[None, :, :], axis=2)
            big = gate_m * 1e3
            rows, cols = linear_sum_assignment(np.where(cost <= gate_m, cost, big))
            for r, c in zip(rows, cols, strict=True):
                if cost[r, c] <= gate_m:
                    live[r].frames.append(f)
                    live[r].positions.append(d[c])
                    used.add(c)
        for c in range(len(d)):
            if c not in used:
                t = Track(id=len(tracks), frames=[f], positions=[d[c]])
                tracks.append(t)
                live.append(t)
    return tracks


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (n, 4) and every box in b (m, 4), boxes x0, y0, x1, y1."""
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def track_boxes(
    frames: Sequence[ArrayLike],
    dt_s: float = 0.5,
    iou_min: float = 0.3,
    max_gap_s: float = 10.0,
    n_init: int = 3,
) -> tuple[list[Track], int]:
    """Tracks from per-frame boxes (each an (n, 4) array, metres), as SORT and DeepSORT do it
    (Bewley et al. 2016; Wojke et al. 2017): each track's last box is moved by its recent velocity,
    detections are assigned by box overlap (IoU at least iou_min, SORT's 0.3), and a new track is
    kept only if it is matched in each of its first n_init frames (DeepSORT's 3), which stops
    one-off false or merged detections becoming animals. Box overlap scales with animal size and
    keeps tightly packed neighbours apart better than centre distance does. Returns the kept tracks
    (positions are box centres, for watch_count) and how many tentative tracks were dropped."""
    if dt_s <= 0 or not 0 < iou_min < 1 or max_gap_s < 0 or n_init < 1:
        raise ValueError("bad tracker settings")
    max_gap = int(round(max_gap_s / dt_s))
    born: list[Track] = []
    failed: set[int] = set()  # indices into born: missed during their first n_init frames
    live: list[tuple[int, list[np.ndarray]]] = []  # index into born, and the track's boxes
    for f, det in enumerate(frames):
        d = np.asarray(det, float).reshape(-1, 4)
        survivors = []
        for k, bx in live:
            t = born[k]
            age = f - t.first  # frames since birth, before this one
            if age <= n_init and len(t.frames) < age:
                failed.add(k)  # a miss while still tentative
            elif f - t.last <= max_gap + 1:
                survivors.append((k, bx))
        live = survivors
        used = set()
        if live and len(d):
            pred = []
            for k, bx in live:
                t = born[k]
                shift = t.predict(f, dt_s) - t.positions[-1]
                pred.append(bx[-1] + np.r_[shift, shift])
            iou = _iou(np.array(pred), d)
            rows, cols = linear_sum_assignment(np.where(iou >= iou_min, 1 - iou, 1e3))
            for r, c in zip(rows, cols, strict=True):
                if iou[r, c] >= iou_min:
                    k, bx = live[r]
                    born[k].frames.append(f)
                    born[k].positions.append((d[c, :2] + d[c, 2:]) / 2)
                    bx.append(d[c])
                    used.add(c)
        for c in range(len(d)):
            if c not in used:
                born.append(Track(id=len(born), frames=[f], positions=[(d[c, :2] + d[c, 2:]) / 2]))
                live.append((len(born) - 1, [d[c]]))
    n_frames = len(frames)
    for k, t in enumerate(born):  # still tentative when the frames ran out
        if len(t.frames) < min(n_init, n_frames - t.first):
            failed.add(k)
    kept = [t for k, t in enumerate(born) if k not in failed]
    return kept, len(failed)


def _confirmed(t: Track, min_frames: int, min_share: float) -> bool:
    span = t.last - t.first + 1
    return len(t.frames) >= min_frames and len(t.frames) / span >= min_share


def watch_count(
    tracks: Sequence[Track],
    n_frames: int,
    min_frames: int = 5,
    min_share: float = 0.2,
    merge_m: float = 1.5,
) -> dict[str, object]:
    """One count and range for the watched spot (see the module docstring for the rules)."""
    conf = sorted(
        (t for t in tracks if _confirmed(t, min_frames, min_share)), key=lambda t: t.first
    )
    # merge identity switches: greedy in time order, each track continued by at most one later
    # track that starts after it ends, close to its end, and never shares a frame with the chain
    chains: list[list[Track]] = []
    for t in conf:
        best, best_d = None, merge_m
        for ch in chains:
            tail = ch[-1]
            if tail.last >= t.first:
                continue
            dist = float(np.linalg.norm(tail.positions[-1] - t.positions[0]))
            if dist <= best_d:
                best, best_d = ch, dist
        if best is None:
            chains.append([t])
        else:
            best.append(t)
    present = np.zeros(n_frames, int)
    for ch in chains:
        seen = sorted({f for t in ch for f in t.frames})
        present[seen] += 1
    return {
        "count": len(chains),
        "low": int(present.max()) if n_frames else 0,
        "high": len(conf),
        "tracks_total": len(tracks),
        "tracks_confirmed": len(conf),
        "switches_merged": len(conf) - len(chains),
    }
