"""ICAERUS grazing-cows dataset v2.

Institut de l'Elevage; Zenodo 10.5281/zenodo.11048412; CC BY 4.0.

Layout as found in the archive (checked 2026-11-18; it differs from the readme in places)::

    Cattle_drone_images/<farm>/JPGImages/<flight>/<stem>.JPG
    Cattle_drone_images/<farm>/YOLO_1.1 or YOLO1.1/<flight>/[obj_train_data/]<stem>.txt
    Cattle_drone_images/<farm>/Pascal_VOC1.1/<flight>/<stem>.xml     (not for every farm)

Findings that shape this loader (docs/data.md):

* YOLO labels exist for every image; PASCAL VOC labels for only 1,079 of 1,385 (none for
  Other_farms). Where both exist they agree to within 1 pixel. YOLO is therefore the label source.
* Three label folders carry a different flight name from their image folder (for example
  Mauron images in DJI_202306071624_028, labels in DJI_202306051814_028). Image file names hold
  a capture timestamp and are unique within a farm, so labels are matched by (farm, file name)
  and the flight is taken from the image folder.
* Some flights define a "calf" class, but no box uses it: every box is class 0, "cow".

`root` is the folder that holds the farm folders (Cattle_drone_images).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from herdsight.data.survey import IMAGE_SUFFIXES, SurveyImage, read_survey_image

SOURCE = "icaerus_v2"
FARMS = ("Derval", "Jalogny", "Mauron", "Other_farms")


@dataclass
class FarmIndex:
    images: dict[str, Path] = field(default_factory=dict)  # stem -> image path
    labels: dict[str, list[Path]] = field(default_factory=dict)  # stem -> YOLO label files
    names: dict[Path, list[str]] = field(default_factory=dict)  # label file -> class names


def _class_names(label_file: Path, label_root: Path) -> list[str]:
    """obj.names in the label file's flight folder, if any; otherwise class 0 is 'cow'."""
    for d in (label_file.parent, label_file.parent.parent):
        if d == label_root.parent:
            break
        f = d / "obj.names"
        if f.is_file():
            return f.read_text(encoding="utf-8").split()
    return ["cow"]


def index_farms(root: Path | str) -> dict[str, FarmIndex]:
    root = Path(root)
    out: dict[str, FarmIndex] = {}
    for farm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        idx = FarmIndex()
        for img_dir in farm_dir.glob("JPGImages"):
            for p in img_dir.rglob("*"):
                if p.suffix.lower() in IMAGE_SUFFIXES:
                    if p.stem in idx.images:
                        raise ValueError(f"duplicate image name in {farm_dir.name}: {p.stem}")
                    idx.images[p.stem] = p
        for yolo_dir in farm_dir.glob("YOLO*"):
            for t in yolo_dir.rglob("*.txt"):
                if t.name == "train.txt":
                    continue
                idx.labels.setdefault(t.stem, []).append(t)
                idx.names[t] = _class_names(t, yolo_dir)
        out[farm_dir.name] = idx
    return out


def read_yolo(path: Path, width: int, height: int) -> tuple[np.ndarray, list[int]]:
    """YOLO rows (class cx cy w h, normalised) -> pixel boxes (x0, y0, x1, y1) and class ids."""
    boxes, cls = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        c, cx, cy, w, h = int(parts[0]), *(float(v) for v in parts[1:5])
        boxes.append(
            [
                (cx - w / 2) * width,
                (cy - h / 2) * height,
                (cx + w / 2) * width,
                (cy + h / 2) * height,
            ]
        )
        cls.append(c)
    return np.asarray(boxes, float).reshape(-1, 4), cls


def iter_images(
    root: Path | str, with_metadata: bool = True
) -> Iterator[tuple[SurveyImage, str | None]]:
    """Yield (SurveyImage with boxes, problem) for every image, sorted by farm, flight and name.

    `problem` is None or a short reason the labels could not be used (the image is then yielded
    without boxes). With with_metadata=False only the image size is read (fast)."""
    for farm, idx in index_farms(root).items():
        for stem, img_path in sorted(idx.images.items(), key=lambda kv: (kv[1].parent.name, kv[0])):
            meta = {"source": SOURCE, "farm": farm, "flight": img_path.parent.name}
            if with_metadata:
                s = read_survey_image(img_path, **meta)
            else:
                with Image.open(img_path) as im:
                    w, h = im.size
                s = SurveyImage(path=img_path, width=w, height=h, **meta)
            files = idx.labels.get(stem, [])
            if not files:
                yield s, "no label file"
                continue
            parsed = [read_yolo(f, s.width, s.height) for f in files]
            if any(not np.allclose(p[0], parsed[0][0]) for p in parsed[1:]):
                yield s, "conflicting label files"
                continue
            boxes, cls = parsed[0]
            names = idx.names[files[0]]
            unknown = [c for c in cls if c >= len(names)]
            if unknown:
                yield s, f"class id {unknown[0]} not in {names}"
                continue
            yield s.with_boxes(boxes), None


def unmatched_labels(root: Path | str) -> dict[str, list[str]]:
    """Label files whose image is not in the archive, per farm (reported in the inventory)."""
    return {
        farm: sorted(set(idx.labels) - set(idx.images))
        for farm, idx in index_farms(root).items()
        if set(idx.labels) - set(idx.images)
    }


def default_root() -> Path:
    """The unpacked archive in the storage folder."""
    from herdsight.utils.paths import storage_path

    return storage_path("raw", "icaerus_grazing_cows_v2", "unpacked", "Cattle_drone_images")
