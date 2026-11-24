"""The take-off clearance rule (herdsight.mission.launch)."""

import numpy as np
import pytest

from herdsight.mission.launch import distance_to_camp_m, require_launch_clearance
from herdsight.sim.herds import Camp

CAMP = Camp(np.array([[35, -90], [285, -90], [285, 90], [35, 90]], float), "test")


@pytest.mark.parametrize(
    "point,expected",
    [
        ((0, 0), 35.0),  # beside an edge
        ((0, 130), np.hypot(35, 40)),  # beyond a corner
        ((100, 0), 0.0),  # inside the camp
        ((35, 0), 0.0),  # on the fence
    ],
)
def test_distance_to_camp(point, expected):
    assert distance_to_camp_m(CAMP, point) == pytest.approx(expected)


def test_launch_far_enough_is_accepted():
    assert require_launch_clearance(CAMP, (0, 0), 30.0) == pytest.approx(35.0)


@pytest.mark.parametrize("point", [(10, 0), (100, 0)])
def test_launch_too_close_or_inside_is_refused(point):
    with pytest.raises(ValueError):
        require_launch_clearance(CAMP, point, 30.0)
