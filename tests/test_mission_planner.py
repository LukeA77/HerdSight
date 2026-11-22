"""Camp survey planner (python/herdsight/mission/planner.py; Phase 5 plan, step 2)."""

from dataclasses import replace

import numpy as np
import pytest

from herdsight.geo.dem import Dem
from herdsight.herd.messages import banned_words
from herdsight.mission.planner import PlannerSettings, PlanRefusedError, choose_height, plan_survey
from herdsight.sim.herds import Camp

UK = PlannerSettings.from_config("uk")


def covered(plan, pts):
    """How many photo footprints contain each point."""
    c, s = np.cos(plan.heading_rad), np.sin(plan.heading_rad)
    rot = np.array([[c, s], [-s, c]])
    across, along = plan.footprint_m
    d = (pts[:, None, :] - plan.photos[None, :, :2]) @ rot.T  # (P, N, 2) in the line frame
    inside = (np.abs(d[..., 0]) <= along / 2) & (np.abs(d[..., 1]) <= across / 2)
    return inside.sum(1)


def grid_in(camp, step=5.0):
    lo, hi = camp.polygon.min(0), camp.polygon.max(0)
    xx, yy = np.meshgrid(np.arange(lo[0], hi[0], step), np.arange(lo[1], hi[1], step))
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    return pts[camp.contains(pts)]


def test_height_from_the_pixel_budget_and_limits():
    h, gsd = choose_height(UK)
    assert h == pytest.approx(UK.focal_px * UK.smallest_animal_m / UK.pixels_needed)
    assert UK.survey_min_m <= h <= UK.max_height_m and gsd == pytest.approx(h / UK.focal_px)
    assert choose_height(replace(UK, pixels_needed=10.0))[0] == UK.max_height_m  # capped
    with pytest.raises(PlanRefusedError) as e:
        choose_height(replace(UK, pixels_needed=100.0))
    assert "safe height" in str(e.value) and not banned_words(str(e.value))


@pytest.mark.parametrize(
    "camp",
    [
        Camp.rectangle(400, 250, "field"),
        Camp(
            np.array([[0, 0], [300, 0], [300, 120], [120, 120], [120, 350], [0, 350]], float), "L"
        ),
        Camp(np.array([[0, 0], [500, 80], [420, 330], [60, 260]], float), "skew"),
    ],
)
def test_every_part_of_the_camp_is_photographed_with_overlap(camp):
    plan = plan_survey(camp, UK)
    n = covered(plan, grid_in(camp))
    assert n.min() >= 1
    assert np.median(n) >= 2  # overlap for duplicate removal
    lines = np.diff(plan.photos[:, :2], axis=0)
    gaps = np.linalg.norm(lines, axis=1)
    assert gaps.min() > 0
    assert plan.photo_spacing_m / plan.speed >= UK.min_photo_interval_s - 1e-9


def test_terrain_is_followed_within_the_limits_or_refused():
    camp = Camp.rectangle(400, 300)
    hill = Dem.from_function(
        400, 300, 5.0, lambda x, y: 25 * np.exp(-((x - 200) ** 2 + (y - 150) ** 2) / 80**2)
    )
    plan = plan_survey(camp, UK, dem=hill, smooth_m=30.0)
    agl = plan.photos[:, 2] - hill.height(plan.photos[:, :2])
    assert agl.min() >= UK.survey_min_m and agl.max() <= UK.max_height_m
    assert np.ptp(plan.photos[:, 2]) > 10  # it climbed over the hill
    cliff = Dem.from_function(400, 300, 5.0, lambda x, y: np.where(x > 200, 100.0, 0.0))
    with pytest.raises(PlanRefusedError) as e:
        plan_survey(camp, UK, dem=cliff, smooth_m=200.0)
    assert not banned_words(str(e.value))


def test_large_camp_is_split_into_sorties_within_battery_and_sight():
    s = PlannerSettings.from_config("south_africa")
    camp = Camp.rectangle(3000, 2700, "large")  # about 810 ha
    plan = plan_survey(camp, s)
    ids = sorted(i for so in plan.sorties for i in so.photo_ids)
    assert ids == list(range(plan.n_photos))  # every photo in exactly one sortie
    for so in plan.sorties:
        assert so.time_s <= s.endurance_s * (1 - s.reserve)
        assert np.linalg.norm(so.photos[:, :2] - so.launch, axis=1).max() <= s.max_distance_m
    assert len(plan.sorties) > 1
    small = plan_survey(Camp.rectangle(300, 200), UK, launch=np.array([0.0, 0.0]))
    assert len(small.sorties) == 1 and np.array_equal(small.sorties[0].launch, [0.0, 0.0])


def test_plan_export_converts_to_px4_frame():
    import yaml

    from herdsight.mission.planner import plan_to_dict

    plan = plan_survey(Camp.rectangle(300, 200), UK)
    d = yaml.safe_load(yaml.safe_dump(plan_to_dict(plan)))
    p = d["sorties"][0]["photos"][0]
    assert p["ned"] == [p["enu"][1], p["enu"][0], -p["enu"][2]]
    assert sum(len(s["photos"]) for s in d["sorties"]) == plan.n_photos
    assert d["height_agl_m"] == pytest.approx(plan.height_agl)
