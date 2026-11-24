"""Sorting the unlabelled ICAERUS raw images (Phase 2, step C5).

Only file names and each photo's date and GPS are used. The order of the rules matters: the test
farm is checked first, so a photo that could belong to it is never used for anything.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

EXCLUSION_RADIUS_KM = 2.0


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0088 * math.asin(min(1.0, math.sqrt(a)))


@dataclass(frozen=True)
class Known:
    """A photo of the labelled dataset."""

    name: str
    farm: str
    date: str  # YYYY-MM-DD
    lat: float
    lon: float
    fold: str  # fold0, fold1, fold2 or test


@dataclass(frozen=True)
class Sorted:
    outcome: str
    farm: str | None = None
    fold: str | None = None


def sort_raw_image(
    name: str,
    date: str | None,
    lat: float | None,
    lon: float | None,
    known: Iterable[Known],
    radius_km: float = EXCLUSION_RADIUS_KM,
) -> Sorted:
    """One of: excluded_test, excluded_no_metadata, already_labelled, same_group, new_day,
    new_place. A photo without a date or GPS cannot be shown to be away from the test farm."""
    rows = list(known)
    test = [k for k in rows if k.fold == "test"]
    if any(k.name == name for k in test):
        return Sorted("excluded_test")
    if date is None or lat is None or lon is None:
        return Sorted("excluded_no_metadata")
    if any(k.date == date for k in test):
        return Sorted("excluded_test")
    if any(distance_km(lat, lon, k.lat, k.lon) <= radius_km for k in test):
        return Sorted("excluded_test")
    developed = [k for k in rows if k.fold != "test"]
    same_name = next((k for k in developed if k.name == name), None)
    if same_name is not None:
        return Sorted("already_labelled", same_name.farm, same_name.fold)
    near = [k for k in developed if distance_km(lat, lon, k.lat, k.lon) <= radius_km]
    if not near:
        return Sorted("new_place")
    same_day = next((k for k in near if k.date == date), None)
    if same_day is not None:
        return Sorted("same_group", same_day.farm, same_day.fold)
    return Sorted("new_day", near[0].farm)
