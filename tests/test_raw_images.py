"""Sorting rules for the unlabelled raw images (herdsight.data.raw_images)."""

import pytest

from herdsight.data.raw_images import Known, distance_km, sort_raw_image

KNOWN = [
    Known("dev.JPG", "Mauron", "2023-06-01", 48.00, -2.00, "fold1"),
    Known("test.JPG", "Other_farms", "2023-07-01", 46.00, 4.00, "test"),
]


def test_distance_of_one_degree_of_latitude():
    assert distance_km(48.0, 0.0, 49.0, 0.0) == pytest.approx(111.2, abs=0.2)


@pytest.mark.parametrize(
    "name,date,lat,lon",
    [
        ("test.JPG", "2024-01-01", 50.0, 1.0),  # same file name as a test photo
        ("x.JPG", "2023-07-01", 50.0, 1.0),  # taken on a test-farm date
        ("x.JPG", "2024-01-01", 46.01, 4.00),  # within 2 km of a test photo
        ("dev.JPG", "2023-07-01", 48.0, -2.0),  # test rule wins over a labelled match
    ],
)
def test_anything_linked_to_the_test_farm_is_excluded(name, date, lat, lon):
    assert sort_raw_image(name, date, lat, lon, KNOWN).outcome == "excluded_test"


@pytest.mark.parametrize("date,lat,lon", [(None, 50.0, 1.0), ("2024-01-01", None, None)])
def test_missing_metadata_is_excluded(date, lat, lon):
    assert sort_raw_image("x.JPG", date, lat, lon, KNOWN).outcome == "excluded_no_metadata"


def test_labelled_photo_is_recognised():
    out = sort_raw_image("dev.JPG", "2023-06-01", 48.0, -2.0, KNOWN)
    assert (out.outcome, out.fold) == ("already_labelled", "fold1")


def test_same_farm_and_day_joins_that_fold():
    out = sort_raw_image("x.JPG", "2023-06-01", 48.001, -2.0, KNOWN)
    assert (out.outcome, out.farm, out.fold) == ("same_group", "Mauron", "fold1")


def test_same_farm_on_a_new_day():
    out = sort_raw_image("x.JPG", "2024-02-02", 48.001, -2.0, KNOWN)
    assert (out.outcome, out.farm, out.fold) == ("new_day", "Mauron", None)


def test_new_place():
    assert sort_raw_image("x.JPG", "2024-02-02", 45.0, 1.0, KNOWN).outcome == "new_place"
