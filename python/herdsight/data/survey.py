"""One record type for every survey still, from public datasets or a real farm flight (D-4).

A real flight produces a folder of geotagged stills: no labels, but camera, GPS, height and
attitude in each file's EXIF and (for DJI drones) XMP metadata. Public datasets add labels. Both
become `SurveyImage`, so every later stage reads the same thing.

Metadata read:

* EXIF: camera make and model, image size, focal length (mm and 35 mm equivalent), capture time,
  GPS latitude, longitude and altitude.
* DJI XMP (`drone-dji:` namespace): relative altitude (above the take-off point), absolute
  altitude, gimbal and flight yaw, pitch and roll. Other makers' XMP is ignored; missing fields
  stay None and are never guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import ExifTags, Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}

_DJI_FIELDS = {
    "RelativeAltitude": "rel_alt_m",
    "AbsoluteAltitude": "xmp_abs_alt_m",
    "GimbalYawDegree": "gimbal_yaw_deg",
    "GimbalPitchDegree": "gimbal_pitch_deg",
    "GimbalRollDegree": "gimbal_roll_deg",
    "FlightYawDegree": "flight_yaw_deg",
    "FlightPitchDegree": "flight_pitch_deg",
    "FlightRollDegree": "flight_roll_deg",
}


@dataclass(frozen=True)
class SurveyImage:
    """A survey still with its metadata and, if labelled, its upright boxes in pixels."""

    path: Path
    width: int
    height: int
    source: str = "unknown"
    farm: str | None = None
    flight: str | None = None
    camera_make: str | None = None
    camera_model: str | None = None
    focal_length_mm: float | None = None
    focal_length_35mm: float | None = None
    time: str | None = None  # EXIF DateTimeOriginal, camera local time, "YYYY:MM:DD HH:MM:SS"
    lat: float | None = None
    lon: float | None = None
    gps_alt_m: float | None = None
    rel_alt_m: float | None = None
    xmp_abs_alt_m: float | None = None
    gimbal_yaw_deg: float | None = None
    gimbal_pitch_deg: float | None = None
    gimbal_roll_deg: float | None = None
    flight_yaw_deg: float | None = None
    flight_pitch_deg: float | None = None
    flight_roll_deg: float | None = None
    boxes_xyxy: NDArray[np.float64] | None = field(default=None, repr=False)

    @property
    def labelled(self) -> bool:
        return self.boxes_xyxy is not None

    @property
    def count(self) -> int | None:
        return None if self.boxes_xyxy is None else int(self.boxes_xyxy.shape[0])

    def with_boxes(self, boxes_xyxy: NDArray[np.float64]) -> SurveyImage:
        return replace(self, boxes_xyxy=np.asarray(boxes_xyxy, float).reshape(-1, 4))


def _rational(v: Any) -> float:
    return float(v[0]) / float(v[1]) if isinstance(v, tuple) else float(v)


def _dms_to_deg(dms: Any, ref: str | None) -> float | None:
    if not dms or len(dms) != 3:
        return None
    deg = _rational(dms[0]) + _rational(dms[1]) / 60 + _rational(dms[2]) / 3600
    return -deg if ref in ("S", "W") else deg


def parse_dji_xmp(xmp: bytes | str) -> dict[str, float]:
    """DJI `drone-dji:` fields from an XMP packet, as attributes or elements."""
    text = xmp.decode("utf-8", "replace") if isinstance(xmp, bytes) else xmp
    out: dict[str, float] = {}
    for key, name in _DJI_FIELDS.items():
        m = re.search(
            rf'drone-dji:{key}\s*=\s*"([^"]+)"|<drone-dji:{key}>([^<]+)</drone-dji:{key}>', text
        )
        if m:
            try:
                out[name] = float((m.group(1) or m.group(2)).strip())
            except ValueError:
                continue
    return out


def _read_xmp(path: Path, max_bytes: int = 2_000_000) -> bytes | None:
    """The XMP packet from the start of the file (DJI writes it near the start)."""
    with path.open("rb") as f:
        head = f.read(max_bytes)
    start = head.find(b"<x:xmpmeta")
    if start < 0:
        return None
    end = head.find(b"</x:xmpmeta>", start)
    return head[start : end + len(b"</x:xmpmeta>")] if end > 0 else None


def read_survey_image(path: Path | str, **labels: Any) -> SurveyImage:
    """Read one still's metadata. Extra keyword arguments (source, farm, flight) are stored."""
    path = Path(path)
    with Image.open(path) as im:
        width, height = im.size
        exif = im.getexif()
    base = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
    sub = {ExifTags.TAGS.get(k, k): v for k, v in exif.get_ifd(ExifTags.IFD.Exif).items()}
    gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in exif.get_ifd(ExifTags.IFD.GPSInfo).items()}

    def _num(v: Any) -> float | None:
        try:
            return None if v is None else _rational(v)
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    gps_alt = _num(gps.get("GPSAltitude"))
    if gps_alt is not None and gps.get("GPSAltitudeRef") in (1, b"\x01"):
        gps_alt = -gps_alt
    xmp = _read_xmp(path)
    dji = parse_dji_xmp(xmp) if xmp else {}
    f35 = sub.get("FocalLengthIn35mmFilm")
    return SurveyImage(
        path=path,
        width=int(width),
        height=int(height),
        camera_make=(str(base["Make"]).strip("\x00 ") if "Make" in base else None),
        camera_model=(str(base["Model"]).strip("\x00 ") if "Model" in base else None),
        focal_length_mm=_num(sub.get("FocalLength")),
        focal_length_35mm=float(f35) if f35 else None,
        time=str(sub.get("DateTimeOriginal") or base.get("DateTime") or "") or None,
        lat=_dms_to_deg(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef")),
        lon=_dms_to_deg(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef")),
        gps_alt_m=gps_alt,
        **dji,
        **labels,
    )


def load_folder(folder: Path | str, source: str = "flight") -> list[SurveyImage]:
    """Every image in a folder of survey stills (a real flight: no labels), sorted by name."""
    folder = Path(folder)
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return [read_survey_image(p, source=source, flight=folder.name) for p in files]
