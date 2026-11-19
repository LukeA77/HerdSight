"""PASCAL VOC annotation files (one XML file per image, upright boxes in pixels).

Downloaded label files are untrusted input, so they are parsed with defusedxml.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from defusedxml import ElementTree
from numpy.typing import NDArray


@dataclass(frozen=True)
class VocAnnotation:
    filename: str | None
    width: int | None
    height: int | None
    boxes_xyxy: NDArray[np.float64]  # (N, 4), VOC's 1-based corners converted to 0-based
    names: tuple[str, ...]


def _int(text: str | None) -> int | None:
    try:
        return int(float(text)) if text is not None else None
    except ValueError:
        return None


def parse_voc(path: Path | str, keep: Iterable[str] | None = None) -> VocAnnotation:
    """Read one VOC file. `keep` limits objects to these class names (case-insensitive).

    VOC pixel coordinates are 1-based and inclusive; they are converted to 0-based continuous
    coordinates (x_min - 1, y_min - 1, x_max, y_max) so a box's width is x_max - x_min."""
    root = ElementTree.parse(str(path)).getroot()
    size = root.find("size")
    width = _int(size.findtext("width")) if size is not None else None
    height = _int(size.findtext("height")) if size is not None else None
    wanted = {k.lower() for k in keep} if keep is not None else None
    boxes, names = [], []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip()
        if wanted is not None and name.lower() not in wanted:
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        x0, y0, x1, y1 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append([x0 - 1.0, y0 - 1.0, x1, y1])
        names.append(name)
    arr = np.asarray(boxes, float).reshape(-1, 4)
    return VocAnnotation(root.findtext("filename"), width, height, arr, tuple(names))
