"""A survey flight over a simulated camp, and home base's count from it, without ROS
(Phase 5 plan, step 5; D-9, D-11, D-13).

`build_world` makes a camp with a moving herd (adults and calves with measured sizes), trees and
fixed objects that cause false detections. `fly` flies a plan over it in time: the herd moves
(and may react to the drone) between stills; each still actually taken sees the animals in its
footprint, with the share of each animal that is visible (frame edges, trees). Home base then
sees only what a real flight would give it: detections drawn from the error model fitted on real
images, each still's geotag with an error drawn from real flights, and, for each overlapping
pair, a registration outcome drawn from real pairs. `count_flight` links animals between
overlapping stills the way the Phase 1 duplicate-removal method does (registered pairs gated at
half a median animal length, geotag pairs at 3 m; one-to-one pairing; identities joined without
two members from one still) and counts the identities. `run_survey` adds the interval from the
real-flight calibration, the herd register and the result record.

Simulation results, never field results. Registration is modelled here, never tested.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from herdsight.count.identities import constrained_identities
from herdsight.count.link import match_points
from herdsight.herd import messages
from herdsight.herd.alerts import Rule
from herdsight.herd.sim import ErrorModel, Survey, measured_flight_error
from herdsight.mission.planner import Plan
from herdsight.sim.errors import DetectionModel, GeotagModel, RegistrationModel
from herdsight.sim.herds import Camp, Herd, HerdSettings, place_herd, settings_from_config
from herdsight.utils.config import REPO_ROOT, load_config

GEOTAG_GATE_M = 3.0  # as scripts/link_animals.py (GEOTAG_PAIR_GATE_M)
REGISTRATION_GATE_SHARE = 0.5  # of the median animal length, as scripts/link_animals.py
LEVELS = (0.68, 0.9, 0.95)
SORTIE_GAP_S = 300.0  # battery swap between sorties


def _interp_quantiles(q: dict[str, float], rng: np.random.Generator, n: int) -> NDArray:
    p = np.array([float(k) / 100 for k in q])
    v = np.array(list(q.values()))
    order = np.argsort(p)
    return np.interp(rng.random(n), p[order], v[order])


@dataclass
class World:
    camp: Camp
    herd: Herd
    length_m: NDArray[np.float64]  # per animal, seen from above
    calf: NDArray[np.bool_]
    trees: NDArray[np.float64]  # (T, 3): x, y, canopy radius
    under_canopy: float
    objects: NDArray[np.float64]  # (O, 2) fixed objects that cause false detections
    det: DetectionModel

    @property
    def median_adult_m(self) -> float:
        adults = self.length_m[~self.calf]
        return float(np.median(adults)) if len(adults) else float(np.median(self.length_m))


def build_world(
    camp: Camp,
    n_animals: int,
    rng: np.random.Generator,
    herd_settings: HerdSettings | None = None,
    calf_share: float | None = None,
    trees_per_ha: float | None = None,
) -> World:
    cfg = load_config("sim_worlds")

    def u(key: str) -> float:
        lo, hi = cfg.require(key)
        return float(rng.uniform(lo, hi))

    s = herd_settings or settings_from_config(rng)
    herd = place_herd(camp, n_animals, s, rng)
    adult_q = json.loads((REPO_ROOT / cfg.require("adult_length_source")).read_text())
    length = _interp_quantiles(adult_q["long_side_m_quantiles"], rng, herd.n)
    share = u("calf_share") if calf_share is None else calf_share
    calf = rng.random(herd.n) < share
    lo_c, hi_c = cfg.require("calf_length_m")
    length[calf] = rng.uniform(lo_c, hi_c, calf.sum())
    ha = camp.area_m2() / 1e4
    n_trees = rng.poisson((u("trees_per_ha") if trees_per_ha is None else trees_per_ha) * ha)
    lo_r, hi_r = cfg.require("tree_canopy_radius_m")
    trees = np.array(
        [[*camp.random_point(rng), rng.uniform(lo_r, hi_r)] for _ in range(n_trees)]
    ).reshape(-1, 3)
    det = DetectionModel.load(p_view=u("false_object_p_view"))
    n_obj = rng.poisson(det.object_density_per_ha(rng) * ha)
    objects = np.array([camp.random_point(rng) for _ in range(n_obj)]).reshape(-1, 2)
    return World(camp, herd, length, calf, trees, u("visible_under_canopy"), objects, det)


@dataclass
class Still:
    index: int
    t: float
    centre: NDArray[np.float64]  # true photo centre (x, y)
    gsd: float
    geotag_error: NDArray[np.float64]  # (2,)
    seen: NDArray[np.int64]  # animals in view
    detected: NDArray[np.int64]  # animal index per detection, -1 for a false detection
    points: NDArray[np.float64]  # detections in ground metres, before the geotag error
    footprint: tuple[float, float] = (0.0, 0.0)  # across, along (metres)
    kind: str = "survey"  # survey | herd


@dataclass
class Flight:
    stills: list[Still]
    planned: int
    coverage: float
    heading: float
    footprint: tuple[float, float]
    true_count: int
    reacted: int
    versions: dict[str, str] = field(default_factory=dict)
    herd_modes: int = 0
    false_triggers: int = 0
    min_descent_clearance_m: float | None = None  # nearest animal where the drone descended


def _in_footprint(
    pts: NDArray, centre: NDArray, heading: float, fp: tuple[float, float]
) -> tuple[NDArray, NDArray]:
    """(inside mask, distance to the footprint edge in metres, negative outside)."""
    c, s = np.cos(heading), np.sin(heading)
    d = (pts - centre) @ np.array([[c, -s], [s, c]])  # columns: along, across
    across, along = fp
    margin = np.minimum(along / 2 - np.abs(d[:, 0]), across / 2 - np.abs(d[:, 1]))
    return margin >= 0, margin


def _herds_in_view(pos: NDArray, seen: NDArray, link_m: float, min_animals: int) -> list[NDArray]:
    """Single-linkage groups (link_m) of the animals in view with at least min_animals."""
    if len(seen) < min_animals:
        return []
    from scipy.cluster.hierarchy import fcluster, linkage

    lab = fcluster(linkage(pos[seen], "single"), link_m, "distance")
    return [seen[lab == k] for k in np.unique(lab) if (lab == k).sum() >= min_animals]


def fly(
    world: World,
    plan: Plan,
    rng: np.random.Generator,
    stop_after: int | None = None,
    new_farm: bool = False,
    herd_mode: bool = False,
    force_trigger: bool = False,
) -> Flight:
    """Fly the plan in time over the moving herd. `stop_after`: the drone turns home after
    this many stills (battery, wind or a failure). `herd_mode`: herds found by the onboard
    trigger (S7 error model) are photographed from above in one still (D-7), descending only
    where no animal is within the clearance radius (D-6). `force_trigger`: every herd in view
    triggers (a test case)."""
    det, geo, herd = world.det, GeotagModel.load(), world.herd
    cfg = load_config("mission")
    s7 = json.loads((REPO_ROOT / cfg.require("herd_trigger_source")).read_text())
    link_m = float(s7["definition"].split("within ")[1].split(" m")[0])
    p_trig, p_false = s7["herd_recall_grouped"], s7["false_triggers_per_image"]
    hm_min_h = float(cfg.require("herd_mode_height_m"))
    hm_min_n = int(cfg.require("herd_mode_min_animals"))
    clearance = float(cfg.require("animal_clearance_radius_m"))
    climb = float(cfg.require("climb_speed_m_s"))
    focal = plan.height_agl / plan.gsd_m
    flight_effect = det.flight_effect(rng, new_farm=new_farm)
    animal_effect = det.animal_effects(herd.n, rng)
    sigma = geo.flight_sigma(rng)
    clock = {"t": 0.0}
    stills: list[Still] = []
    reacted = np.zeros(herd.n, bool)
    inspected: set[int] = set()
    counters = {"herd": 0, "false": 0}
    clearances: list[float] = []

    def advance(seconds: float, x: float, y: float, h: float) -> None:
        nonlocal reacted
        end = clock["t"] + seconds
        while clock["t"] < end:
            step = min(1.0, end - clock["t"])
            reacted |= herd.step(step, rng, drone=(x, y, h))
            clock["t"] += step

    def capture(
        index: int, centre: NDArray, gsd: float, fp: tuple[float, float], kind: str
    ) -> Still:
        inside, margin = _in_footprint(herd.pos, centre, plan.heading_rad, fp)
        seen = np.flatnonzero(inside)
        length = world.length_m[seen]
        visible = np.clip(margin[seen] / length + 0.5, 0.0, 1.0)  # cut by the frame edge
        if len(world.trees):
            d = np.linalg.norm(herd.pos[seen, None] - world.trees[None, :, :2], axis=2)
            visible = np.where(
                (d < world.trees[None, :, 2]).any(1), visible * world.under_canopy, visible
            )
        dist = np.linalg.norm(herd.pos[seen, None] - herd.pos[None], axis=2)
        dist[np.arange(len(seen)), seen] = np.inf
        crowded = (dist < length[:, None]).any(1)
        found = det.found(length / gsd, crowded, visible, flight_effect, rng, animal_effect[seen])
        ids = seen[found]
        pts = herd.pos[ids] + det.position_noise_m(world.length_m[ids], rng)
        if len(world.objects):
            o_in, _ = _in_footprint(world.objects, centre, plan.heading_rad, fp)
            o_hit = np.flatnonzero(o_in & (rng.random(len(world.objects)) < det.p_view))
            pts = np.vstack([pts, world.objects[o_hit]])
            ids = np.concatenate([ids, np.full(len(o_hit), -1)])
        return Still(
            index, clock["t"], centre, gsd, rng.normal(0, sigma, 2), seen, ids, pts, fp, kind
        )

    def inspect(centre: NDArray, members: NDArray | None, back_to: NDArray) -> None:
        """Herd mode: descend clear of the animals, move over the herd, one still, climb."""
        if members is not None and len(members):
            extent = 2 * float(np.max(np.linalg.norm(herd.pos[members] - centre, axis=1))) + 10.0
        else:
            extent = 20.0
        # lowest height (not below the herd-mode floor) whose still covers the herd with margin
        h = max(hm_min_h, plan.height_agl * 1.2 * extent / plan.footprint_m[1])
        if h >= plan.height_agl:
            return  # the herd already fits a survey still: no herd mode
        # descent point: away from the herd towards the survey track, clear of every animal
        away = back_to - centre
        away = away / np.linalg.norm(away) if np.linalg.norm(away) > 1e-6 else np.array([1.0, 0])
        r = extent / 2 + clearance
        for _ in range(30):
            descent = centre + away * r
            near = float(np.min(np.linalg.norm(herd.pos - descent, axis=1)))
            if near >= clearance:
                break
            r += 10.0
        clearances.append(near)
        advance(np.linalg.norm(descent - back_to) / plan.speed, *descent, plan.height_agl)
        advance((plan.height_agl - h) / climb, *descent, h)
        advance(np.linalg.norm(centre - descent) / plan.speed, *centre, h)
        scale = h / plan.height_agl
        fp = (plan.footprint_m[0] * scale, plan.footprint_m[1] * scale)
        stills.append(capture(-1, centre.copy(), h / focal, fp, "herd"))
        advance((plan.height_agl - h) / climb, *centre, plan.height_agl)
        advance(np.linalg.norm(back_to - centre) / plan.speed, *back_to, plan.height_agl)

    taken = 0
    for so in plan.sorties:
        prev = so.launch
        for pid, (x, y, _z) in zip(so.photo_ids, so.photos, strict=True):
            if stop_after is not None and taken >= stop_after:
                break
            centre = np.array([x, y])
            advance(np.linalg.norm(centre - prev) / plan.speed, x, y, plan.height_agl)
            prev = centre
            still = capture(pid, centre, plan.gsd_m, plan.footprint_m, "survey")
            stills.append(still)
            taken += 1
            if not herd_mode:
                continue
            herds = _herds_in_view(herd.pos, still.seen, link_m, hm_min_n)
            for members in herds:
                if len(inspected & set(members.tolist())) > len(members) / 2:
                    continue  # this herd has had its herd-mode still
                if force_trigger or rng.random() < p_trig:
                    counters["herd"] += 1
                    inspected |= set(members.tolist())
                    inspect(herd.pos[members].mean(0), members, centre)
            if not herds and rng.random() < p_false:
                counters["false"] += 1
                inspect(centre + rng.uniform(-20, 20, 2), None, centre)
        else:
            clock["t"] += SORTIE_GAP_S
            continue
        break
    survey = [s.centre for s in stills if s.kind == "survey"]
    coverage = _coverage(world.camp, survey, plan)
    return Flight(
        stills=stills,
        planned=plan.n_photos,
        coverage=coverage,
        heading=plan.heading_rad,
        footprint=plan.footprint_m,
        true_count=herd.n,
        reacted=int(reacted.sum()),
        versions={
            "detection": det.version,
            "geotag": geo.version,
            "registration": RegistrationModel.load().version,
            "matcher": "phase1-registration-greedy (D-27, D-30)",
            "herd_trigger": f"S7 error model ({cfg.require('herd_trigger_source')})",
        },
        herd_modes=counters["herd"],
        false_triggers=counters["false"],
        min_descent_clearance_m=min(clearances) if clearances else None,
    )


def _coverage(camp: Camp, centres: list[NDArray], plan: Plan, step: float = 5.0) -> float:
    lo, hi = camp.polygon.min(0), camp.polygon.max(0)
    xx, yy = np.meshgrid(np.arange(lo[0], hi[0], step), np.arange(lo[1], hi[1], step))
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    pts = pts[camp.contains(pts)]
    if not len(pts) or not centres:
        return 0.0
    seen = np.zeros(len(pts), bool)
    for c in centres:
        seen |= _in_footprint(pts, c, plan.heading_rad, plan.footprint_m)[0]
    return float(seen.mean())


def count_flight(flight: Flight, median_animal_m: float, rng: np.random.Generator) -> dict:
    """Home base: link detections between overlapping stills and count identities.

    Merge rule for herd mode (Phase 5 plan, step 7): survey detections inside a herd-mode
    still's footprint (as placed by the geotags) and within herd_mode_window_s of it are
    replaced by the herd-mode still's detections."""
    reg = RegistrationModel.load()
    window = float(load_config("mission").require("herd_mode_window_s"))
    st = flight.stills
    points = [s.points.copy() for s in st]
    for h in (s for s in st if s.kind == "herd"):
        for k, s in enumerate(st):
            if s.kind != "survey" or abs(s.t - h.t) > window or not len(points[k]):
                continue
            est = points[k] + s.geotag_error
            inside, _ = _in_footprint(est, h.centre + h.geotag_error, flight.heading, h.footprint)
            points[k] = points[k][~inside]
    counts = [len(p) for p in points]
    links = []
    accepted = fallback = 0
    c, s_ = np.cos(flight.heading), np.sin(flight.heading)
    rot = np.array([[c, -s_], [s_, c]])
    for i in range(len(st)):
        for j in range(i + 1, len(st)):
            d = (st[j].centre - st[i].centre) @ rot
            fa = (st[i].footprint[1] + st[j].footprint[1]) / 2
            fc = (st[i].footprint[0] + st[j].footprint[0]) / 2
            if abs(d[0]) >= fa or abs(d[1]) >= fc or not counts[i] or not counts[j]:
                continue
            pi = points[i] + st[i].geotag_error
            residual = reg.outcome(rng)
            if residual is not None:  # registered: j's points brought into i's frame
                accepted += 1
                pj = points[j] + st[i].geotag_error + rng.normal(0, residual * st[i].gsd, 2)
                gate = REGISTRATION_GATE_SHARE * median_animal_m
            else:
                fallback += 1
                pj = points[j] + st[j].geotag_error
                gate = GEOTAG_GATE_M
            for a, b, dist in match_points(pi, pj, gate):
                links.append(((i, a), (j, b), dist))
    ids = constrained_identities(counts, links)
    return {
        "count": len(ids),
        "detections": int(sum(counts)),
        "pairs_registered": accepted,
        "pairs_geotag": fallback,
        "herd_mode_stills": sum(s.kind == "herd" for s in st),
    }


def real_flight_intervals(count: int, rng: np.random.Generator) -> tuple[dict, dict]:
    """Intervals from the count error measured on real flights (never the simulation's own)."""
    bias, sd = measured_flight_error()
    q = ErrorModel(bias=bias, sd=sd, name="measured").calibrate(LEVELS, rng)
    lo = {lv: max(0.0, count - qq * (count + 1)) for lv, qq in q.items()}
    hi = {lv: count + qq * (count + 1) for lv, qq in q.items()}
    return lo, hi


def run_survey(
    world: World,
    plan: Plan,
    rng: np.random.Generator,
    expected: int,
    farm_id: str = "sim",
    stop_after: int | None = None,
    new_farm: bool = False,
    rule: Rule | None = None,
    herd_mode: bool = False,
) -> dict:
    """Fly, count at home base, and write the result record for the camp."""
    flight = fly(world, plan, rng, stop_after=stop_after, new_farm=new_farm, herd_mode=herd_mode)
    res = count_flight(flight, world.median_adult_m, rng)
    record = flight_record(flight, res, world.camp.name, expected, farm_id, rule)
    return {"record": record, "flight": flight, "count": res, "true_count": flight.true_count}


def flight_record(
    flight: Flight,
    res: dict,
    camp: str,
    expected: int,
    farm_id: str = "sim",
    rule: Rule | None = None,
) -> dict:
    """The result record for one counted flight, with its range from real-flight calibration."""
    rule = rule or Rule("recount", level=0.9)
    lo, hi = real_flight_intervals(res["count"], np.random.default_rng(0))
    s = Survey(
        day=0.0,
        camp=camp,
        true=flight.true_count,
        count=res["count"],
        lo=lo,
        hi=hi,
        dense=False,
    )
    short = hi[rule.level] < expected
    big = short and expected - hi[rule.level] >= rule.big
    alert = {"kind": "missing", "checks": 1, "other_camp": None} if big else None
    return messages.survey_record(
        s,
        farm_id=farm_id,
        expected=expected,
        rule=rule,
        alert=alert,
        count_error_model="real flights (results/baselines/dfine_s3.json, registration)",
        pipeline_version="; ".join(f"{k}: {v}" for k, v in flight.versions.items()),
        coverage=flight.coverage,
        recount_pending=short and not big,
    )
