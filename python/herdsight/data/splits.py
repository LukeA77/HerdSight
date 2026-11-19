"""Splits by farm and survey date, never by image or by single flight (brief section 8.2).

Rules, fixed before any model is run:

1. **Test farm.** The farm whose share of all labelled boxes is closest to `test_share` (20%);
   ties broken alphabetically. The whole farm is held out until Phase 7.
2. **Groups.** The remaining flights are grouped by farm and survey date: flights flown over the
   same farm on the same day see the same herd in the same light, so they stay together (D-25).
3. **Folds.** Groups are dealt into `n_folds` cross-validation folds (D-26): taken in the order
   of a hash of their names (fixed, but unrelated to content), each goes to the fold currently
   furthest below an equal share of boxes. Groups with no boxes are spread by image count.

Only counts of boxes and images are used, never model results.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

DEFAULT_FOLDS = 3
_DATE = re.compile(r"(\d{8})")


@dataclass(frozen=True)
class FlightStats:
    farm: str
    flight: str
    n_images: int
    n_boxes: int
    date: str | None = None  # YYYYMMDD; read from the flight name if not given

    @property
    def group(self) -> str:
        d = self.date
        if d is None:
            m = _DATE.search(self.flight)
            d = m.group(1) if m else self.flight  # no date: the flight is its own group
        return f"{self.farm}/{d}"


def choose_test_farm(flights: list[FlightStats], test_share: float = 0.2) -> str:
    total = sum(f.n_boxes for f in flights)
    if total == 0:
        raise ValueError("no labelled boxes")
    by_farm: dict[str, int] = {}
    for f in flights:
        by_farm[f.farm] = by_farm.get(f.farm, 0) + f.n_boxes
    return min(sorted(by_farm), key=lambda farm: abs(by_farm[farm] / total - test_share))


def _hash_order(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def _assign(groups: dict[str, int], shares: Mapping[str, float]) -> dict[str, str]:
    total = sum(groups.values()) or 1
    got = dict.fromkeys(shares, 0.0)
    out: dict[str, str] = {}
    for g in sorted(groups, key=_hash_order):
        # The split furthest below its target; ties go to the first split listed.
        split = max(shares, key=lambda s: shares[s] - got[s] / total)
        out[g] = split
        got[split] += groups[g]
    return out


def make_splits(
    flights: list[FlightStats], n_folds: int = DEFAULT_FOLDS, test_share: float = 0.2
) -> dict[str, object]:
    """Assign every flight to the test farm or to one of `n_folds` folds."""
    if n_folds < 2:
        raise ValueError("need at least two folds")
    keys = [f"{f.farm}/{f.flight}" for f in flights]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate flight")
    test_farm = choose_test_farm(flights, test_share)
    rest = [f for f in flights if f.farm != test_farm]
    boxes: dict[str, int] = {}
    images: dict[str, int] = {}
    for f in rest:
        boxes[f.group] = boxes.get(f.group, 0) + f.n_boxes
        images[f.group] = images.get(f.group, 0) + f.n_images
    shares = {f"fold{k}": 1.0 / n_folds for k in range(n_folds)}
    group_fold = _assign({g: b for g, b in boxes.items() if b > 0}, shares)
    group_fold |= _assign({g: images[g] for g, b in boxes.items() if b == 0}, shares)
    assignment = {f"{f.farm}/{f.flight}": "test" for f in flights if f.farm == test_farm}
    assignment |= {f"{f.farm}/{f.flight}": group_fold[f.group] for f in rest}
    summary: dict[str, dict[str, int]] = {}
    for f in flights:
        s = summary.setdefault(assignment[f"{f.farm}/{f.flight}"], {"groups": 0, "flights": 0})
        s["flights"] += 1
        s["images"] = s.get("images", 0) + f.n_images
        s["boxes"] = s.get("boxes", 0) + f.n_boxes
    for fold in group_fold.values():
        summary[fold]["groups"] += 1
    return {
        "rule": "test = whole farm with box share closest to test_share; other flights grouped by "
        "farm and survey date; groups in hash order to the fold furthest below an equal share of "
        "boxes (empty groups by images)",
        "test_farm": test_farm,
        "test_share": test_share,
        "n_folds": n_folds,
        "groups": dict(sorted(group_fold.items())),
        "flights": dict(sorted(assignment.items())),
        "summary": dict(sorted(summary.items())),
    }


def sample_images(names: list[str], share: float, seed: int) -> list[str]:
    """A fixed random sample of images (sorted names, seeded permutation, first ceil(share * n)).

    Used where compute forces a subset (the CPU zero-shot baseline); every method compared on
    the subset is scored on exactly the same images."""
    if not 0 < share <= 1:
        raise ValueError("share must be in (0, 1]")
    ordered = sorted(names)
    k = math.ceil(share * len(ordered))
    pick = np.random.default_rng(seed).permutation(len(ordered))[:k]
    return sorted(ordered[i] for i in pick)
