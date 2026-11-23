"""Home base's count from one finished Gazebo flight (Phase 5, step 8; D-11).

`count_run` reads a run of sim/bridge/run_flight.sh with BBOX=1 and works from the stills actually
taken. For each still it finds the bounding-box camera frame nearest in simulation time; frames can
arrive late, so their receive time is never used for pairing. The frame says which animals were in
view, where, and how much of each was visible. The detection error model fitted on real images
then decides which are detected, adds position noise and adds false detections from fixed objects;
a geotag error is drawn from real flights. A frame can be up to 0.1 s from the trigger (0.8 m at
survey speed), which a real camera's image and geotag are not, so each frame is placed from the
camera's own position at that frame plus PX4's estimate error at the trigger.
The count, range and result record come from the same code as the flight model without ROS.

The camera is an ideal stabilised camera, so the tilt of the drone is not in these results.
Simulation only; never a field result.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from herdsight.herd.alerts import Rule
from herdsight.sim.errors import DetectionModel, GeotagModel, RegistrationModel
from herdsight.sim.flight import Flight, Still, _in_footprint, count_flight, flight_record
from herdsight.sim.handoff import HERD_STILL_HEIGHT_SHARE, make_bundle, read_stills

IMAGE_PX = (660, 495)  # as sim/bridge/add_survey_camera.py
MAX_FRAME_GAP_S = 0.15  # the camera runs at 5 Hz, so the nearest frame is within 0.1 s
# A frame shows the camera where it was this long before the frame's time stamp. Measured once, on
# a calibration flight (not one of the checked flights), as the delay that best lines up the same
# animals seen on passes flown in opposite directions.
RENDER_DELAY_S = 0.1


def read_frames(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Rows of bbox.jsonl by sensor; a line cut short when the flight stopped is skipped."""
    out: dict[str, list[dict[str, Any]]] = {"visible": [], "full": [], "clock": [], "pose": []}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[row["sensor"]].append(row)
    for rows in out.values():
        rows.sort(key=lambda r: r["sim_s"])
    return out


def _nearest(rows: list[dict[str, Any]], times: NDArray, t: float) -> dict[str, Any] | None:
    if not len(times):
        return None
    k = int(np.abs(times - t).argmin())
    return rows[k] if abs(times[k] - t) <= MAX_FRAME_GAP_S else None


def box_to_ground(
    box: list[float],
    centre: NDArray[np.float64],
    heading: float,
    footprint: tuple[float, float],
    image_px: tuple[int, int] = IMAGE_PX,
) -> NDArray[np.float64]:
    """Ground position of a box centre. The top of the image is along the heading and its left
    is 90 degrees anticlockwise of it; `footprint` is across (image width), along (height)."""
    w, h = image_px
    across, along = footprint
    forward = (h / 2 - (box[1] + box[3]) / 2) / h * along
    left = (w / 2 - (box[0] + box[2]) / 2) / w * across
    c, s = np.cos(heading), np.sin(heading)
    return centre + np.array([forward * c - left * s, forward * s + left * c])


def _area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _pixel_area(box: list[float]) -> float:
    """Area of a visible box, whose corners are whole pixels and both inside the box."""
    return max(0.0, box[2] - box[0] + 1) * max(0.0, box[3] - box[1] + 1)


def _view(
    visible: dict[str, Any], full: dict[str, Any] | None, n_animals: int
) -> tuple[NDArray, NDArray, NDArray]:
    """(animal index, visible share, box used for the position) of the labelled animals seen."""
    whole = {int(b[0]): b[1:] for b in full["boxes"]} if full else {}
    ids, share, boxes = [], [], []
    for b in visible["boxes"]:
        label = int(b[0])
        if not 1 <= label <= n_animals:
            continue
        seen_box = b[1:]
        box = whole.get(label, seen_box)
        ids.append(label - 1)
        share.append(min(1.0, _pixel_area(seen_box) / _area(box)) if _area(box) > 0 else 0.0)
        boxes.append(box)
    return (
        np.array(ids, int),
        np.array(share, float),
        np.array(boxes, float).reshape(-1, 4),
    )


def count_run(
    run: Path,
    expected: int | None = None,
    seed: int = 0,
    rule: Rule | None = None,
    farm_id: str = "sim",
) -> dict[str, Any]:
    """Count one finished flight; returns the bundle, count, result record and checks."""
    bundle = make_bundle(run)
    plan = yaml.safe_load((run / "plan.yaml").read_text(encoding="utf-8"))
    with (run / "world_state.pkl").open("rb") as fh:  # written by sim/bridge/make_world.py
        state = pickle.load(fh)
    with (run / "herd_state.pkl").open("rb") as fh:
        herd = pickle.load(fh)
    length_m, calf, objects = state["length_m"], state["calf"], state["objects"]
    n_animals = len(length_m)
    expected = n_animals if expected is None else expected
    frames = read_frames(run / "bbox.jsonl")
    if len(frames["clock"]) < 2:
        raise ValueError(f"{run}: no simulation clock in bbox.jsonl")
    wall = np.array([r["wall_s"] for r in frames["clock"]])
    sim = np.array([r["sim_s"] for r in frames["clock"]])
    order = np.argsort(wall)
    wall, sim = wall[order], sim[order]
    t_vis = np.array([r["sim_s"] for r in frames["visible"]])
    t_full = np.array([r["sim_s"] for r in frames["full"]])
    t_pose = np.array([r["sim_s"] for r in frames["pose"]])
    xy_drone = np.array([r["drone"][:2] for r in frames["pose"]], float).reshape(-1, 2)
    xy_camera = np.array([r["camera"][:2] for r in frames["pose"]], float).reshape(-1, 2)

    def at(xy: NDArray, t: float) -> NDArray:
        return np.array([np.interp(t, t_pose, xy[:, 0]), np.interp(t, t_pose, xy[:, 1])])

    rng = np.random.default_rng(seed)
    det = DetectionModel.load(p_view=float(state["p_view"]))
    geo = GeotagModel.load()
    flight_effect = det.flight_effect(rng)
    animal_effect = det.animal_effects(n_animals, rng)
    sigma = geo.flight_sigma(rng)
    height = float(plan["height_agl_m"])
    heading = float(plan["heading_enu_rad"])
    base_fp = (float(plan["footprint_m"][0]), float(plan["footprint_m"][1]))
    base_gsd = float(plan["gsd_m"])

    stills: list[Still] = []
    without_frame = 0
    animals_in_view: set[int] = set()
    for raw in sorted(read_stills(run / "stills.csv"), key=lambda r: r["t_s"]):
        t_sim = float(np.interp(raw["t_s"], wall, sim))
        vis = _nearest(frames["visible"], t_vis, t_sim)
        if vis is None:
            without_frame += 1
            continue
        x, y, z = raw["enu_m"]
        centre = np.array([x, y], float)
        if len(t_pose) >= 2:  # the camera at this frame, with PX4's estimate error at the trigger
            centre = (
                at(xy_camera, float(vis["sim_s"]) - RENDER_DELAY_S) + centre - at(xy_drone, t_sim)
            )
        kind = "survey" if z >= HERD_STILL_HEIGHT_SHARE * height else "herd"
        scale = z / height
        fp = (base_fp[0] * scale, base_fp[1] * scale)
        gsd = base_gsd * scale
        ids, share, boxes = _view(vis, _nearest(frames["full"], t_full, t_sim), n_animals)
        animals_in_view |= set(ids.tolist())
        ground = np.array([box_to_ground(b.tolist(), centre, heading, fp) for b in boxes]).reshape(
            -1, 2
        )
        lengths = length_m[ids]
        if len(ids):
            dist = np.linalg.norm(ground[:, None] - ground[None], axis=2)
            np.fill_diagonal(dist, np.inf)
            crowded = (dist < lengths[:, None]).any(1)
        else:
            crowded = np.zeros(0, bool)
        found = det.found(lengths / gsd, crowded, share, flight_effect, rng, animal_effect[ids])
        hit = ids[found]
        points = ground[found] + det.position_noise_m(length_m[hit], rng)
        if len(objects):
            inside, _ = _in_footprint(objects, centre, heading, fp)
            o_hit = np.flatnonzero(inside & (rng.random(len(objects)) < det.p_view))
            points = np.vstack([points, objects[o_hit]])
            hit = np.concatenate([hit, np.full(len(o_hit), -1)])
        stills.append(
            Still(
                raw["photo_id"] if kind == "survey" else -1,
                t_sim,
                centre,
                gsd,
                rng.normal(0, sigma, 2),
                ids,
                hit,
                points,
                fp,
                kind,
            )
        )
    flight = Flight(
        stills=stills,
        planned=bundle["photos_planned"],
        coverage=float(bundle["coverage"]),
        heading=heading,
        footprint=base_fp,
        true_count=herd.n,
        reacted=0,
        versions={
            "detection_source": "Gazebo bounding-box camera (ideal stabilised camera)",
            "detection": det.version,
            "geotag": geo.version,
            "registration": RegistrationModel.load().version,
            "matcher": "phase1-registration-greedy (D-27, D-30)",
        },
        herd_modes=bundle["herd_mode_stills"],
    )
    adults = length_m[~calf] if (~calf).any() else length_m
    res = count_flight(flight, float(np.median(adults)), rng)
    record = flight_record(flight, res, str(bundle["camp"]), expected, farm_id, rule)
    pose = frames["pose"]
    gap = (
        np.linalg.norm(
            np.array([r["drone"] for r in pose]) - np.array([r["camera"] for r in pose]), axis=1
        )
        if pose
        else np.zeros(0)
    )
    return {
        "bundle": bundle,
        "count": res,
        "record": record,
        "checks": {
            "stills_without_a_camera_frame": without_frame,
            "animals_in_view_of_a_still": len(animals_in_view),
            "animals_in_world": n_animals,
            "camera_to_drone_gap_median_m": float(np.median(gap)) if len(gap) else None,
        },
    }
