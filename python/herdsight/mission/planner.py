"""Camp survey planner (Phase 5 plan, step 2; D-6, D-7, D-19). Python reference.

Plans a survey of one camp as geotagged stills on parallel lines:
  1. Height: the highest height at which the smallest animal (a calf) still covers the required
     pixels: height = focal length in pixels x required ground sample distance (GSD), capped by
     the rules profile's ceiling. If that is below the welfare floor, the mission is refused with
     a plain reason; a limit is never broken quietly.
  2. Lines: parallel to the camp's long axis, spaced so neighbouring lines' photos overlap by
     side_overlap; photos along a line overlap by forward_overlap; the camera's long side lies
     across the lines. Lines are clipped to the camp and flown back and forth.
  3. Terrain: each photo point is flown at the chosen height above a smoothed terrain model; the
     true height above the unsmoothed terrain must stay within the welfare floor and the ceiling.
  4. Sorties: the route is cut into sorties that each fit the battery (with a reserve) and keep
     every photo point within the rules profile's distance of that sortie's take-off point.
Settings come from configs/mission.yaml. Distances in metres, local east-north frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from herdsight.geo.dem import Dem
from herdsight.geo.nadir import focal_px_from_35mm
from herdsight.sim.herds import Camp
from herdsight.utils.config import load_config


class PlanRefusedError(Exception):
    """The mission cannot be flown within the limits; the message is plain English."""


@dataclass(frozen=True)
class PlannerSettings:
    width_px: int
    height_px: int
    focal_px: float
    smallest_animal_m: float
    pixels_needed: float
    survey_min_m: float
    max_height_m: float
    max_distance_m: float
    forward_overlap: float
    side_overlap: float
    speed: float
    min_photo_interval_s: float
    turn_time_s: float
    endurance_s: float
    reserve: float
    climb_speed: float
    profile: str

    @staticmethod
    def from_config(profile: str = "uk", **overrides: float) -> PlannerSettings:
        c = load_config("mission")
        cam = c.require("design_camera")
        rules = c.require("rules_profiles")
        if profile not in rules:
            raise ValueError(f"no rules profile {profile!r}; known: {sorted(rules)}")
        r = rules[profile]
        s = dict(
            width_px=int(cam["width_px"]),
            height_px=int(cam["height_px"]),
            focal_px=focal_px_from_35mm(
                float(cam["focal_35mm"]), cam["width_px"], cam["height_px"]
            ),
            smallest_animal_m=float(c.require("smallest_animal_length_m")),
            pixels_needed=float(c.require("pixels_for_smallest_animal")),
            survey_min_m=float(c.require("survey_height_min_m")),
            max_height_m=float(r["max_height_m"]),
            max_distance_m=float(r["max_distance_m"]),
            forward_overlap=float(c.require("forward_overlap")),
            side_overlap=float(c.require("side_overlap")),
            speed=float(c.require("cruise_speed_m_s")),
            min_photo_interval_s=float(c.require("min_photo_interval_s")),
            turn_time_s=float(c.require("turn_time_s")),
            endurance_s=60 * float(c.require("endurance_min")),
            reserve=float(c.require("battery_reserve_share")),
            climb_speed=float(c.require("climb_speed_m_s")),
            profile=profile,
        )
        s.update(overrides)
        return PlannerSettings(**s)


@dataclass
class Sortie:
    launch: NDArray[np.float64]  # (2,) take-off and landing point
    photos: NDArray[np.float64]  # (K, 3) x, y, height above sea level of the drone
    photo_ids: list[int]
    lines: int
    time_s: float


@dataclass
class Plan:
    camp: str
    profile: str
    height_agl: float
    gsd_m: float
    footprint_m: tuple[float, float]  # across the lines, along the lines
    line_spacing_m: float
    photo_spacing_m: float
    speed: float
    heading_rad: float  # direction of the survey lines
    photos: NDArray[np.float64]  # (N, 3) every photo point
    sorties: list[Sortie] = field(default_factory=list)

    @property
    def n_photos(self) -> int:
        return len(self.photos)


def choose_height(s: PlannerSettings) -> tuple[float, float]:
    """(height above ground, GSD): the highest allowed height that still resolves the smallest
    animal. Raises PlanRefusedError if that is below the welfare floor."""
    gsd_needed = s.smallest_animal_m / s.pixels_needed
    height = min(s.focal_px * gsd_needed, s.max_height_m)
    if height < s.survey_min_m:
        need = s.focal_px * gsd_needed
        raise PlanRefusedError(
            f"To see calves clearly this camera would have to fly at {need:.0f} m, lower than the "
            f"safe height for animals ({s.survey_min_m:.0f} m). Use a camera with more zoom."
        )
    return height, height / s.focal_px


def _long_axis(poly: NDArray[np.float64]) -> float:
    """Direction (radians) of the camp's longest extent: the minimum-area bounding rectangle's
    long side, over the polygon's edge directions."""
    best = None
    for (x0, y0), (x1, y1) in zip(poly, np.roll(poly, -1, axis=0), strict=True):
        a = np.arctan2(y1 - y0, x1 - x0)
        r = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        q = poly @ r.T
        w, h = np.ptp(q[:, 0]), np.ptp(q[:, 1])
        area = w * h
        if best is None or area < best[0]:
            best = (area, a if w >= h else a + np.pi / 2)
    return float(best[1])


def _clip_line(poly: NDArray[np.float64], y: float) -> list[tuple[float, float]]:
    """Segments [x_start, x_end] where the horizontal line at y lies inside the polygon."""
    xs = []
    for (x0, y0), (x1, y1) in zip(poly, np.roll(poly, -1, axis=0), strict=True):
        if (y0 > y) != (y1 > y):
            xs.append(x0 + (y - y0) * (x1 - x0) / (y1 - y0))
    xs.sort()
    return [(xs[k], xs[k + 1]) for k in range(0, len(xs) - 1, 2)]


def plan_survey(
    camp: Camp,
    s: PlannerSettings,
    dem: Dem | None = None,
    smooth_m: float = 50.0,
    launch: NDArray[np.float64] | None = None,
) -> Plan:
    height, gsd = choose_height(s)
    across = s.width_px * gsd  # camera long side across the lines
    along = s.height_px * gsd
    spacing = across * (1 - s.side_overlap)
    step = along * (1 - s.forward_overlap)
    speed = min(s.speed, step / s.min_photo_interval_s)
    heading = _long_axis(camp.polygon)
    c, sn = np.cos(heading), np.sin(heading)
    rot = np.array([[c, sn], [-sn, c]])  # world -> line frame (lines along +x)
    q = camp.polygon @ rot.T
    y_lo, y_hi = q[:, 1].min(), q[:, 1].max()
    n_lines = max(1, int(np.ceil((y_hi - y_lo) / spacing)))
    # line footprints reach across/2 > spacing/2 either side, so the camp edges are covered
    ys = (
        np.linspace(y_lo + spacing / 2, y_hi - spacing / 2, n_lines)
        if n_lines > 1
        else [(y_lo + y_hi) / 2]
    )
    # Large camps are cut into blocks along the lines, short enough that the pilot can keep the
    # drone in sight from one spot in each block (about 1.4 times the distance limit, so a
    # block's corners stay within the limit of its centre for blocks up to that wide).
    x_lo, x_hi = q[:, 0].min(), q[:, 0].max()
    block_len = 1.4 * s.max_distance_m
    n_blocks = max(1, int(np.ceil((x_hi - x_lo) / block_len)))
    edges = np.linspace(x_lo, x_hi, n_blocks + 1)
    bands = max(1, int(np.ceil((y_hi - y_lo) / block_len)))
    band_of = np.minimum(
        ((np.asarray(ys) - y_lo) / (y_hi - y_lo + 1e-9) * bands).astype(int), bands - 1
    )
    pts, line_of = [], []
    unit = 0
    for blk in range(n_blocks):
        for band in range(bands):
            rows = [k for k in range(len(ys)) if band_of[k] == band]
            for r_i, k in enumerate(rows):
                segs = []
                for a, b in _clip_line(q, ys[k]):
                    a, b = max(a, edges[blk]), min(b, edges[blk + 1])
                    if b > a:
                        segs.append((a, b))
                if r_i % 2:
                    segs = [(b, a) for a, b in reversed(segs)]
                for a, b in segs:
                    # photo centres from boundary to boundary; footprints reach along/2 beyond
                    n = max(2, int(np.ceil(abs(b - a) / step)) + 1)
                    for x in np.linspace(a, b, n):
                        pts.append((x, ys[k]))
                        line_of.append(unit)
                    unit += 1
    xy = np.array(pts) @ rot  # back to world
    ground = dem.smoothed(smooth_m).height(xy) if dem is not None else np.zeros(len(xy))
    photos = np.column_stack([xy, ground + height])
    if dem is not None:
        true_agl = photos[:, 2] - dem.height(xy)
        if true_agl.min() < s.survey_min_m - 1e-6 or true_agl.max() > s.max_height_m + 1e-6:
            lo, hi = true_agl.min(), true_agl.max()
            raise PlanRefusedError(
                f"The ground in this camp is too uneven to keep a safe and legal height "
                f"(between {lo:.0f} m and {hi:.0f} m above the ground on this route)."
            )
    plan = Plan(
        camp=camp.name,
        profile=s.profile,
        height_agl=height,
        gsd_m=gsd,
        footprint_m=(across, along),
        line_spacing_m=spacing,
        photo_spacing_m=step,
        speed=speed,
        heading_rad=heading,
        photos=photos,
    )
    plan.sorties = _split_sorties(photos, np.array(line_of), s, speed, height, launch)
    return plan


def _sortie_time(
    photos: NDArray[np.float64],
    lines: int,
    s: PlannerSettings,
    speed: float,
    launch: NDArray[np.float64],
    height: float,
) -> float:
    path = np.sum(np.linalg.norm(np.diff(photos[:, :2], axis=0), axis=1)) if len(photos) > 1 else 0
    legs = np.linalg.norm(photos[0, :2] - launch) + np.linalg.norm(photos[-1, :2] - launch)
    return float((path + legs) / speed + lines * s.turn_time_s + 2 * height / s.climb_speed)


def _split_sorties(
    photos: NDArray[np.float64],
    line_of: NDArray[np.int64],
    s: PlannerSettings,
    speed: float,
    height: float,
    launch: NDArray[np.float64] | None,
) -> list[Sortie]:
    """One sortie from `launch` if the whole route fits the battery and every photo is within the
    distance limit of it. Otherwise whole lines are added to a sortie while it fits, each sortie
    taking off from the centre of its photos (the pilot moves between sorties)."""
    budget = s.endurance_s * (1 - s.reserve)

    def make(ids: list[int], lp: NDArray[np.float64] | None = None) -> Sortie | None:
        p = photos[ids]
        lp = p[:, :2].mean(0) if lp is None else np.asarray(lp, float)
        if np.max(np.linalg.norm(p[:, :2] - lp, axis=1)) > s.max_distance_m:
            return None
        n_lines = len(set(line_of[ids].tolist()))
        t = _sortie_time(p, n_lines, s, speed, lp, height)
        return Sortie(lp, p, ids, n_lines, t) if t <= budget else None

    if launch is not None:
        whole = make(list(range(len(photos))), launch)
        if whole is not None:
            return [whole]
    sorties: list[Sortie] = []
    current: list[int] = []
    for line in dict.fromkeys(line_of.tolist()):
        ids = np.flatnonzero(line_of == line).tolist()
        if current and make(current + ids) is not None:
            current += ids
            continue
        if current:
            sorties.append(make(current))
        if make(ids) is None:
            raise PlanRefusedError(
                "One survey line of this camp is too long for one battery or to keep the drone "
                "in sight; split the camp into smaller areas."
            )
        current = ids
    if current:
        sorties.append(make(current))
    return sorties


def plan_to_dict(plan: Plan) -> dict:
    """The plan for the onboard mission (ros2/herdsight_mission reads it as YAML).

    PX4's local frame is north-east-down; the planner works east-north-up. Each photo point is
    given in both, with its height above the take-off point, so the C++ side does no conversion
    of its own."""
    sorties = []
    for so in plan.sorties:
        photos = []
        for pid, (x, y, z) in zip(so.photo_ids, so.photos, strict=True):
            photos.append(
                {
                    "id": int(pid),
                    "enu": [float(x), float(y), float(z)],
                    "ned": [float(y), float(x), -float(z)],
                }
            )
        sorties.append(
            {
                "launch_enu": [float(v) for v in so.launch],
                "photos": photos,
                "time_s": float(so.time_s),
            }
        )
    return {
        "camp": plan.camp,
        "profile": plan.profile,
        "height_agl_m": float(plan.height_agl),
        "gsd_m": float(plan.gsd_m),
        "speed_m_s": float(plan.speed),
        "heading_enu_rad": float(plan.heading_rad),
        # yaw in NED: clockwise from north; ENU heading is anticlockwise from east
        "yaw_ned_rad": float(np.pi / 2 - plan.heading_rad),
        "footprint_m": [float(v) for v in plan.footprint_m],
        "sorties": sorties,
    }
