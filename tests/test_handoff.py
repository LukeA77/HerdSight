import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from herdsight.sim.handoff import camp_coverage, make_bundle, read_stills
from herdsight.sim.herds import Camp

CAMP = Camp.rectangle(100, 100, "t")


def _run(tmp_path, taken, landed=True, herd=False):
    photos = [
        {"id": 0, "enu": [25.0, 50.0, 80.0]},
        {"id": 1, "enu": [75.0, 50.0, 80.0]},
    ]
    plan = {
        "camp": "Test camp",
        "height_agl_m": 80.0,
        "heading_enu_rad": 0.0,
        "footprint_m": [100.0, 50.0],
        "sorties": [{"photos": photos}],
    }
    (tmp_path / "plan.yaml").write_text(yaml.safe_dump(plan), encoding="utf-8")
    with (tmp_path / "herd_state.pkl").open("wb") as fh:
        pickle.dump(SimpleNamespace(camp=CAMP), fh)
    rows = ["this line is not a still"]
    for k in taken:
        x, y, z = photos[k]["enu"]
        rows.append(f"100,5,map,{k},{x},{y},{z},{z},1.57,-1.57,,")
    if herd:
        rows.append("120,0,map,0,50.0,50.0,40.5,40.5,1.57,-1.57,,")
    (tmp_path / "stills.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    text = "INFO Disarmed by landing\n" if landed else "INFO flying\n"
    (tmp_path / "px4.log").write_text(text, encoding="utf-8")
    return tmp_path


def test_read_stills_skips_other_lines(tmp_path):
    run = _run(tmp_path, [0])
    s = read_stills(run / "stills.csv")
    assert len(s) == 1 and s[0]["photo_id"] == 0
    assert s[0]["t_s"] == pytest.approx(100 + 5e-9)


def test_coverage_of_footprints():
    one = camp_coverage(CAMP, np.array([[25.0, 50.0]]), 0.0, (100.0, 50.0))
    two = camp_coverage(CAMP, np.array([[25.0, 50.0], [75.0, 50.0]]), 0.0, (100.0, 50.0))
    assert one == pytest.approx(0.5, abs=0.06)
    assert two == 1.0
    assert camp_coverage(CAMP, np.zeros((0, 2)), 0.0, (100.0, 50.0)) == 0.0


def test_complete_flight(tmp_path):
    b = make_bundle(_run(tmp_path, [0, 1]))
    assert b["outcome"] == "completed" and b["photos_missing"] == []
    assert b["coverage_enough_for_alert"] and b["simulated"]


def test_early_return_cannot_raise_an_alert(tmp_path):
    b = make_bundle(_run(tmp_path, [0]))
    assert b["outcome"] == "ended_early" and b["photos_missing"] == [1]
    assert not b["coverage_enough_for_alert"]


def test_herd_still_is_not_counted_as_a_survey_photo(tmp_path):
    b = make_bundle(_run(tmp_path, [0], herd=True))
    assert b["photos_taken"] == 1 and b["herd_mode_stills"] == 1
    assert b["photos_missing"] == [1]


def test_flight_that_did_not_land(tmp_path):
    assert make_bundle(_run(tmp_path, [0, 1], landed=False))["outcome"] == "not_landed"
