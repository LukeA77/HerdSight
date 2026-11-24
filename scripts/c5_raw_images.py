"""Sort the unlabelled ICAERUS raw images by date and GPS (Phase 2, step C5).

Usage: python scripts/c5_raw_images.py --archive <Cattle_drone_images.zip>

Reads only each photo's file name and metadata header; no pixels are decoded. Rules are in
herdsight.data.raw_images. Writes the per-photo list to
$HERDSIGHT_STORAGE/runs/phase2/c5/raw_images.csv and counts to results/phase2/c5_raw_images.json.
Photos linked to the test farm are listed only as excluded. Nothing is used for training here:
photos from new places or days still need the owner's screening.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import ExifTags, Image

from herdsight.data.raw_images import EXCLUSION_RADIUS_KM, Known, sort_raw_image
from herdsight.utils.config import REPO_ROOT
from herdsight.utils.paths import storage_path

EXIF_IFD, GPS_IFD = 0x8769, 0x8825
DATE_ORIGINAL, DATE_TIME = 36867, 306


def degrees(value, ref) -> float | None:  # noqa: ANN001
    try:
        d, m, s = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    out = d + m / 60 + s / 3600
    return -out if str(ref).upper() in ("S", "W") else out


def read_meta(zf: zipfile.ZipFile, member: str) -> tuple[str | None, float | None, float | None]:
    try:
        with zf.open(member) as fh, Image.open(fh) as im:  # the header only; pixels stay unread
            exif = im.getexif()
            stamp = exif.get_ifd(EXIF_IFD).get(DATE_ORIGINAL) or exif.get(DATE_TIME)
            gps = exif.get_ifd(GPS_IFD)
    except (OSError, ValueError, SyntaxError):
        return None, None, None
    date = str(stamp)[:10].replace(":", "-") if stamp else None
    tags = ExifTags.GPS
    lat = degrees(gps.get(tags.GPSLatitude), gps.get(tags.GPSLatitudeRef))
    lon = degrees(gps.get(tags.GPSLongitude), gps.get(tags.GPSLongitudeRef))
    return date, lat, lon


def known_images() -> list[Known]:
    sp = json.loads((REPO_ROOT / "results" / "splits.json").read_text())["icaerus_v2"]
    out = []
    path = REPO_ROOT / "results" / "data" / "icaerus_images.csv"
    with path.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            fold = sp["flights"].get(f"{r['farm']}/{r['flight']}")
            if fold is None and r["farm"] == "Other_farms":
                fold = "test"
            if fold is None or not r["lat"] or not r["time"]:
                continue
            out.append(
                Known(
                    r["image"],
                    r["farm"],
                    r["time"][:10].replace(":", "-"),
                    float(r["lat"]),
                    float(r["lon"]),
                    fold,
                )
            )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, required=True)
    args = ap.parse_args(argv)
    known = known_images()
    if not any(k.fold == "test" for k in known):
        raise SystemExit("no test-farm reference points: refusing to sort")
    rows = []
    with zipfile.ZipFile(args.archive) as zf:
        members = sorted(
            n
            for n in zf.namelist()
            if n.lower().endswith((".jpg", ".jpeg")) and "__MACOSX" not in n
        )
        for i, member in enumerate(members, 1):
            name = PurePosixPath(member).name
            date, lat, lon = read_meta(zf, member)
            s = sort_raw_image(name, date, lat, lon, known)
            hidden = s.outcome == "excluded_test"
            rows.append(
                {
                    "member": member,
                    "outcome": s.outcome,
                    "farm": s.farm or "",
                    "fold": s.fold or "",
                    "date": "" if hidden else date or "",
                    "lat": "" if hidden or lat is None else round(lat, 6),
                    "lon": "" if hidden or lon is None else round(lon, 6),
                }
            )
            if i % 100 == 0:
                print(f"{i}/{len(members)}", flush=True)
    out = storage_path("runs", "phase2", "c5", create=True) / "raw_images.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    counts = Counter(r["outcome"] for r in rows)
    usable = counts["new_day"] + counts["new_place"]
    summary = {
        "note": "Sorted by file name, date and GPS only. No image is used until the owner has "
        "screened it.",
        "source": "ICAERUS raw cattle images, Zenodo 10.5281/zenodo.8234156, CC BY 4.0",
        "exclusion_radius_km": EXCLUSION_RADIUS_KM,
        "images": len(rows),
        "outcomes": dict(sorted(counts.items())),
        "new_places_or_days_needing_screening": usable,
        "new_days_by_farm": dict(
            sorted(Counter(r["farm"] for r in rows if r["outcome"] == "new_day").items())
        ),
    }
    dest = REPO_ROOT / "results" / "phase2" / "c5_raw_images.json"
    dest.write_bytes((json.dumps(summary, indent=1) + "\n").encode("utf-8"))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
