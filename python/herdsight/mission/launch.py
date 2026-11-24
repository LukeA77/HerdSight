"""Where the drone may take off and land (Phase 5; D-46).

The welfare rule is that the drone only climbs or descends where no animal is within the clearance
radius. Animals move, so a check of where they are now would not hold for the whole flight. A launch
point at least the clearance radius outside the fence holds for every animal inside the camp, for
as long as the fence holds.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from herdsight.sim.herds import Camp


def distance_to_camp_m(camp: Camp, point: ArrayLike) -> float:
    """Distance from a point to the camp's fence; 0 if the point is inside the camp."""
    p = np.asarray(point, float).reshape(2)
    if bool(camp.contains(p)[0]):
        return 0.0
    a = np.asarray(camp.polygon, float)
    b = np.roll(a, -1, axis=0)
    edge = b - a
    t = np.clip(((p - a) * edge).sum(1) / np.maximum((edge * edge).sum(1), 1e-12), 0.0, 1.0)
    nearest = a + t[:, None] * edge
    return float(np.linalg.norm(nearest - p, axis=1).min())


def require_launch_clearance(camp: Camp, launch: ArrayLike, clearance_m: float) -> float:
    """The launch point's distance from the fence; refuses a point closer than the clearance."""
    d = distance_to_camp_m(camp, launch)
    if d < clearance_m:
        raise ValueError(
            f"launch point is {d:.1f} m from the camp fence; the welfare rule needs at least "
            f"{clearance_m:.1f} m"
        )
    return d
