"""Count at home base from a finished Gazebo flight (python/herdsight/sim/home_base.py)."""

import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from jsonschema import Draft202012Validator, FormatChecker

from herdsight.herd import messages
from herdsight.sim.herds import Camp
from herdsight.sim.home_base import IMAGE_PX, _true_ground, box_to_ground, count_run, read_frames

VALIDATOR = Draft202012Validator(
    json.loads(messages.SCHEMA.read_text(encoding="utf-8")), format_checker=FormatChecker()
)
CAMP = Camp.rectangle(100, 100, "t")
WALL_0 = 1000.0


def _run(tmp_path, taken=(0, 1), late_s=0.0):
    """Two stills at 80 m. Animal 1 is in the first still only; frames arrive `late_s` late."""
    photos = [{"id": 0, "enu": [25.0, 50.0, 80.0]}, {"id": 1, "enu": [75.0, 50.0, 80.0]}]
    plan = {
        "camp": "Test camp",
        "height_agl_m": 80.0,
        "heading_enu_rad": 0.0,
        "footprint_m": [100.0, 75.0],
        "gsd_m": 0.02,
        "sorties": [{"photos": photos}],
    }
    (tmp_path / "plan.yaml").write_text(yaml.safe_dump(plan), encoding="utf-8")
    with (tmp_path / "herd_state.pkl").open("wb") as fh:
        pickle.dump(SimpleNamespace(camp=CAMP, n=2), fh)
    with (tmp_path / "world_state.pkl").open("wb") as fh:
        pickle.dump(
            {
                "length_m": np.array([2.5, 2.5]),
                "calf": np.array([False, False]),
                "objects": np.zeros((0, 2)),
                "p_view": 0.5,
            },
            fh,
        )
    rows = [
        f"{int(WALL_0 + 10 * k)},0,map,{k},{photos[k]['enu'][0]},50.0,80.0,80.0,0,-1.57,,"
        for k in taken
    ]
    (tmp_path / "stills.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (tmp_path / "px4.log").write_text("INFO Disarmed by landing\n", encoding="utf-8")
    lines = []
    for t in np.arange(0.0, 21.0, 0.1):
        lines.append({"sensor": "clock", "wall_s": WALL_0 + t, "sim_s": 500.0 + t})
    for t in np.arange(0.0, 21.0, 0.2):
        boxes = [[1, 320, 240, 340, 250]]
        if t < 5:
            boxes.append([2, 100, 100, 120, 110])
        for sensor in ("visible", "full"):
            lines.append(
                {
                    "sensor": sensor,
                    "wall_s": WALL_0 + t + late_s,
                    "sim_s": 500.0 + t,
                    "boxes": boxes,
                }
            )
    (tmp_path / "bbox.jsonl").write_text(
        "\n".join(json.dumps(r) for r in lines) + '\n{"sensor": "vis', encoding="utf-8"
    )
    return tmp_path


def test_box_centre_to_ground_follows_the_image_axes():
    w, h = IMAGE_PX
    centre = np.array([10.0, 20.0])
    fp = (120.0, 90.0)
    middle = [w / 2 - 1, h / 2 - 1, w / 2 + 1, h / 2 + 1]
    assert box_to_ground(middle, centre, 0.0, fp) == pytest.approx(centre)
    top = [w / 2 - 1, 0, w / 2 + 1, 2]
    assert box_to_ground(top, centre, 0.0, fp)[0] == pytest.approx(10 + 45 - 90 / h)
    left = [0, h / 2 - 1, 2, h / 2 + 1]
    assert box_to_ground(left, centre, 0.0, fp)[1] == pytest.approx(20 + 60 - 120 / w)
    # heading east to north: the top of the image now points north
    assert box_to_ground(top, centre, np.pi / 2, fp)[1] == pytest.approx(20 + 45 - 90 / h)


def test_a_line_cut_short_is_skipped(tmp_path):
    frames = read_frames(_run(tmp_path) / "bbox.jsonl")
    assert len(frames["clock"]) == 210 and len(frames["visible"]) == 105


def test_complete_flight_gives_a_valid_simulated_record(tmp_path):
    out = count_run(_run(tmp_path), expected=2)
    assert not list(VALIDATOR.iter_errors(out["record"]))
    assert out["record"]["simulated"] is True
    assert out["checks"]["animals_in_view_of_a_still"] == 2
    assert out["checks"]["stills_without_a_camera_frame"] == 0


def test_frames_are_paired_by_simulation_time_not_arrival(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    on_time = count_run(_run(tmp_path / "a"), expected=2, seed=3)
    late = count_run(_run(tmp_path / "b", late_s=7.0), expected=2, seed=3)
    assert late["count"] == on_time["count"]
    assert late["checks"] == on_time["checks"]


def test_interrupted_flight_cannot_raise_an_alert(tmp_path):
    out = count_run(_run(tmp_path, taken=(0,)), expected=40)
    assert out["bundle"]["outcome"] == "ended_early"
    assert out["record"]["status"] == "unknown" and out["record"]["alerts"] == []


def test_true_ground_interpolates_each_animal_between_log_times():
    times = np.array([0.0, 2.0])
    xy = np.array([[[0.0, 0.0], [10.0, 10.0]], [[2.0, 4.0], [10.0, 10.0]]])
    out = _true_ground(times, xy, np.array([1, 0]), 1.0)
    assert out == pytest.approx(np.array([[10.0, 10.0], [1.0, 2.0]]))
