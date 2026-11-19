"""ICAERUS loader on a synthetic copy of the archive's real layout, and the split rules."""

import numpy as np
import pytest
from PIL import Image

from herdsight.data.icaerus import iter_images, unmatched_labels
from herdsight.data.splits import FlightStats, choose_test_farm, make_splits, sample_images


def _make(root, farm, flight, stems, n_cows=1, label_flight=None, nested=True, size=(64, 48)):
    img_dir = root / farm / "JPGImages" / flight
    lab_dir = root / farm / "YOLO_1.1" / (label_flight or flight)
    txt_dir = lab_dir / "obj_train_data" if nested else lab_dir
    img_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)
    if nested:
        (lab_dir / "obj.names").write_text("cow\ncalf\n")
    for s in stems:
        Image.new("RGB", size).save(img_dir / f"{s}.JPG")
        rows = "".join(f"0 {0.1 + 0.2 * i:.6f} 0.5 0.125 0.25\n" for i in range(n_cows))
        (txt_dir / f"{s}.txt").write_text(rows)
    return txt_dir


def test_loader_reads_yolo_in_both_layouts(tmp_path):
    _make(tmp_path, "Mauron", "DJI_202306211015_001", ["DJI_0001", "DJI_0002"], n_cows=3)
    _make(tmp_path, "Derval", "DJI_202307011200_002", ["DJI_0003"], n_cows=0, nested=False)
    items = list(iter_images(tmp_path))
    assert len(items) == 3 and all(p is None for _, p in items)
    counts = {(s.farm, s.path.stem): s.count for s, _ in items}
    assert counts == {
        ("Derval", "DJI_0003"): 0,
        ("Mauron", "DJI_0001"): 3,
        ("Mauron", "DJI_0002"): 3,
    }
    s = next(s for s, _ in items if s.path.stem == "DJI_0001")
    # cx 0.1, cy 0.5, w 0.125, h 0.25 on a 64 x 48 image
    np.testing.assert_allclose(s.boxes_xyxy[0], [2.4, 18, 10.4, 30])


def test_labels_are_matched_by_name_when_flight_folders_disagree(tmp_path):
    _make(
        tmp_path,
        "Mauron",
        "DJI_202306071624_028",
        ["DJI_A"],
        n_cows=2,
        label_flight="DJI_202306051814_028",
    )
    ((s, problem),) = list(iter_images(tmp_path, with_metadata=False))
    assert problem is None and s.count == 2 and s.flight == "DJI_202306071624_028"
    assert (s.width, s.height) == (64, 48)


def test_missing_labels_and_orphan_labels_are_reported(tmp_path):
    txt_dir = _make(tmp_path, "Jalogny", "DJI_202310011000_003", ["A"])
    (txt_dir / "A.txt").unlink()
    (txt_dir / "ORPHAN.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    ((s, problem),) = list(iter_images(tmp_path, with_metadata=False))
    assert problem == "no label file"
    assert unmatched_labels(tmp_path) == {"Jalogny": ["ORPHAN"]}


def test_unknown_class_id_is_reported(tmp_path):
    txt_dir = _make(tmp_path, "Derval", "DJI_202310011000_004", ["B"])
    (txt_dir / "B.txt").write_text("5 0.5 0.5 0.1 0.1\n")
    ((_, problem),) = list(iter_images(tmp_path, with_metadata=False))
    assert problem.startswith("class id 5")


def _fl(farm, i, boxes, images=10):
    return FlightStats(farm, f"DJI_{i:03d}", images, boxes)


def test_test_farm_is_closest_to_twenty_percent():
    flights = [_fl("A", 1, 50), _fl("B", 2, 22), _fl("C", 3, 28)]
    assert choose_test_farm(flights) == "B"  # 22% is closest to 20%


def test_splits_hold_out_the_whole_test_farm_and_balance_folds():
    flights = [
        FlightStats("Big", f"DJI_2023{1 + i // 28:02d}{1 + i % 28:02d}1200_{i:03d}", 10, 10 + i % 7)
        for i in range(40)
    ]
    flights += [_fl("Small", 100 + i, 12) for i in range(10)]
    flights += [FlightStats("Big", f"DJI_2024010{i}1200_{200 + i}", 30, 0) for i in range(6)]
    s = make_splits(flights)
    assert s["test_farm"] == "Small"
    assert all(v == "test" for k, v in s["flights"].items() if k.startswith("Small/"))
    assert not any(v == "test" for k, v in s["flights"].items() if k.startswith("Big/"))
    boxes = {k: v["boxes"] for k, v in s["summary"].items() if k != "test"}
    total = sum(boxes.values())
    for k in ("fold0", "fold1", "fold2"):
        assert boxes[k] / total == pytest.approx(1 / 3, abs=0.05)
    empty = [f"Big/DJI_2024010{i}1200_{200 + i}" for i in range(6)]
    assert {s["flights"][k] for k in empty} == {"fold0", "fold1", "fold2"}


def test_splits_are_deterministic_and_reject_duplicates():
    flights = [_fl("A", i, 5) for i in range(20)] + [_fl("B", 50, 25)]
    assert make_splits(flights) == make_splits(list(reversed(flights)))
    with pytest.raises(ValueError):
        make_splits(flights + [flights[0]])


def test_sample_images_is_fixed_and_sized():
    names = [f"img{i:03d}" for i in range(40)]
    a = sample_images(names, 0.25, seed=0)
    assert len(a) == 10 and a == sample_images(list(reversed(names)), 0.25, seed=0)
    assert a != sample_images(names, 0.25, seed=1)


def test_flights_from_the_same_farm_and_day_stay_together():
    flights = [
        FlightStats("Big", f"DJI_2023{m:02d}{d:02d}1045_{i:03d}", 10, 10 + i)
        for i, (m, d) in enumerate([(m, d) for m in range(6, 10) for d in (1, 15)] * 2)
    ]
    flights += [FlightStats("Small", "DJI_202301011200_900", 10, 40)]
    s = make_splits(flights)
    by_group = {}
    for f in flights:
        if f.farm == "Big":
            by_group.setdefault(f.group, set()).add(s["flights"][f"{f.farm}/{f.flight}"])
    assert all(len(v) == 1 for v in by_group.values())
    assert len(by_group) == 8
