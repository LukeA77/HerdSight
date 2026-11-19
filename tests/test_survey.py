"""Survey still metadata, using small synthetic JPEGs with known EXIF and DJI XMP."""

import numpy as np
import pytest
from PIL import Image

from herdsight.data.survey import load_folder, parse_dji_xmp, read_survey_image

XMP = (
    b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description '
    b'drone-dji:RelativeAltitude="+100.20" drone-dji:AbsoluteAltitude="+245.5" '
    b'drone-dji:GimbalPitchDegree="-90.00" drone-dji:GimbalYawDegree="+12.3" '
    b'drone-dji:FlightYawDegree="+11.9"/></rdf:RDF></x:xmpmeta>'
)


def _jpeg(path, xmp=XMP, south=False):
    im = Image.new("RGB", (64, 48), (90, 140, 60))
    exif = Image.Exif()
    exif[0x010F] = "DJI"  # Make
    exif[0x0110] = "M3E"  # Model
    ifd = exif.get_ifd(0x8769)
    ifd[0x920A] = 12.29  # FocalLength
    ifd[0xA405] = 24  # FocalLengthIn35mmFilm
    ifd[0x9003] = "2023:06:21 10:15:00"  # DateTimeOriginal
    gps = exif.get_ifd(0x8825)
    gps[1] = "S" if south else "N"
    gps[2] = (47.0, 30.0, 36.0)
    gps[3] = "W"
    gps[4] = (1.0, 45.0, 0.0)
    gps[6] = 245.5
    im.save(path, exif=exif, xmp=xmp)
    return path


def test_reads_exif_gps_and_dji_xmp(tmp_path):
    s = read_survey_image(_jpeg(tmp_path / "a.jpg"), source="test", farm="f", flight="x")
    assert (s.width, s.height) == (64, 48)
    assert (s.camera_make, s.camera_model) == ("DJI", "M3E")
    assert s.focal_length_mm == pytest.approx(12.29) and s.focal_length_35mm == 24
    assert s.time == "2023:06:21 10:15:00"
    assert s.lat == pytest.approx(47.51) and s.lon == pytest.approx(-1.75)
    assert s.gps_alt_m == pytest.approx(245.5)
    assert s.rel_alt_m == pytest.approx(100.2) and s.gimbal_pitch_deg == pytest.approx(-90)
    assert s.gimbal_yaw_deg == pytest.approx(12.3) and s.flight_yaw_deg == pytest.approx(11.9)
    assert s.gimbal_roll_deg is None  # absent fields are never guessed
    assert (s.source, s.farm, s.flight) == ("test", "f", "x")
    assert not s.labelled and s.count is None


def test_southern_latitude_is_negative(tmp_path):
    s = read_survey_image(_jpeg(tmp_path / "a.jpg", south=True))
    assert s.lat == pytest.approx(-47.51)


def test_missing_xmp_leaves_fields_none(tmp_path):
    s = read_survey_image(_jpeg(tmp_path / "a.jpg", xmp=b""))
    assert s.rel_alt_m is None and s.gimbal_pitch_deg is None


def test_xmp_element_form_is_parsed():
    xmp = "<drone-dji:RelativeAltitude>60.1</drone-dji:RelativeAltitude>"
    assert parse_dji_xmp(xmp) == {"rel_alt_m": pytest.approx(60.1)}


def test_with_boxes_and_count(tmp_path):
    s = read_survey_image(_jpeg(tmp_path / "a.jpg")).with_boxes([[1, 2, 3, 4], [5, 6, 7, 8]])
    assert s.labelled and s.count == 2
    np.testing.assert_allclose(s.boxes_xyxy[1], [5, 6, 7, 8])


def test_load_folder_reads_a_real_flight_layout(tmp_path):
    flight = tmp_path / "DJI_202306211015_001"
    flight.mkdir()
    for name in ("DJI_0002.JPG", "DJI_0001.JPG"):
        _jpeg(flight / name)
    (flight / "notes.txt").write_text("not an image")
    imgs = load_folder(flight)
    assert [p.path.name for p in imgs] == ["DJI_0001.JPG", "DJI_0002.JPG"]
    assert all(i.flight == flight.name and not i.labelled for i in imgs)
