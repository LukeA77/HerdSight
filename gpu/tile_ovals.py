"""Animal ovals for training tiles, and the pixel work of coat darkening (Phase 2, dark-coat,
tight-group and outline training). NumPy and Pillow only, so it is tested without D-FINE.

The C6 labels (gpu/runpod/job_c6.sh, results/phase2/c6_sam_labels.json) give each training animal
a rotated box from its SAM outline: centre, length, width and angle, in photo pixels. Seen from
above a cow is close to an oval, so the oval inside that rotated box stands in for its outline.
Training tiles are full-resolution crops named 'farm__flight__stem__x0_y0.jpg'
(scripts/make_training_tiles.py), so a tile's ovals are the photo's ovals moved by (-x0, -y0).

  parse_tile_name   tile file name -> (photo key 'farm/flight/stem', x0, y0)
  tile_ovals        rotated boxes per photo + tile names -> ovals per tile (tile pixels)
  oval_alpha        soft 0..1 mask of one oval over an image
  darken_coats      darken (and part-desaturate) the pixels inside chosen ovals
gpu/make_tile_ovals.py writes the per-tile file; gpu/dfine_ops.py uses the rest on the pod.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

Oval = list[float]  # [cx, cy, length, width, angle_rad] in tile pixels


def parse_tile_name(file_name: str) -> tuple[str, int, int]:
    """'Derval__DJI_x__DJI_y_V__448_0.jpg' -> ('Derval/DJI_x/DJI_y_V', 448, 0)."""
    name = file_name.rsplit("/", 1)[-1]
    if name.lower().endswith(".jpg"):
        name = name[:-4]
    parts = name.split("__")
    if len(parts) != 4:
        raise ValueError(f"not a tile name: {file_name}")
    farm, flight, stem, xy = parts
    x0, y0 = (int(v) for v in xy.split("_"))
    return f"{farm}/{flight}/{stem}", x0, y0


def photo_key(image: str) -> str:
    """C6 label path 'Derval/flight/DJI_..._V.JPG' -> 'Derval/flight/DJI_..._V'."""
    return image.rsplit(".", 1)[0]


def tile_ovals(
    obbs_by_photo: Mapping[str, Sequence[Sequence[float]]],
    tile_names: Iterable[str],
    tile_px: int,
) -> dict[str, list[Oval]]:
    """Ovals whose centre lies inside each tile, in that tile's pixels. Tiles without any are
    left out, so an empty or unlabelled tile is never given an animal."""
    out: dict[str, list[Oval]] = {}
    for name in tile_names:
        key, x0, y0 = parse_tile_name(name)
        rows = []
        for cx, cy, length, width, ang in obbs_by_photo.get(key, ()):
            tx, ty = cx - x0, cy - y0
            if 0 <= tx < tile_px and 0 <= ty < tile_px:
                rows.append([round(tx, 2), round(ty, 2), length, width, ang])
        if rows:
            out[name] = rows
    return out


def oval_alpha(
    w: int, h: int, oval: Sequence[float], shrink: float = 0.95, feather_px: float = 1.5
) -> np.ndarray:
    """0..1 mask (h, w) of the oval inscribed in a rotated box, slightly shrunk so the edge stays
    on the animal, with a soft edge feather_px wide."""
    cx, cy, length, width, ang = oval
    a = max(length * shrink / 2, 1e-3)
    b = max(width * shrink / 2, 1e-3)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    c, s = math.cos(ang), math.sin(ang)
    u = ((xx + 0.5 - cx) * c + (yy + 0.5 - cy) * s) / a
    v = (-(xx + 0.5 - cx) * s + (yy + 0.5 - cy) * c) / b
    r = np.sqrt(u * u + v * v)
    edge = feather_px / max(min(a, b), 1e-3)  # feather width in units of the normalised radius
    return np.clip((1.0 - r) / max(edge, 1e-6) + 0.5, 0.0, 1.0)


def darken_coats(
    rgb: np.ndarray, ovals: Sequence[Sequence[float]], factor: float, desaturate: float
) -> np.ndarray:
    """Pixels inside the ovals: brightness times `factor` (0..1) and colour pulled `desaturate`
    (0..1) of the way to grey, blended through each oval's soft mask. Outside is unchanged."""
    if not 0 < factor <= 1 or not 0 <= desaturate <= 1:
        raise ValueError("factor in (0, 1], desaturate in [0, 1]")
    img = rgb.astype(np.float32)
    h, w = img.shape[:2]
    alpha = np.zeros((h, w), np.float32)
    for o in ovals:
        alpha = np.maximum(alpha, oval_alpha(w, h, o))
    grey = img.mean(axis=2, keepdims=True)
    dark = (img * (1 - desaturate) + grey * desaturate) * factor
    out = img * (1 - alpha[..., None]) + dark * alpha[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def oval_box(oval: Sequence[float]) -> list[float]:
    """Axis-aligned box [x0, y0, x1, y1] around the oval inscribed in a rotated box."""
    cx, cy, length, width, ang = oval
    a, b = length / 2, width / 2
    c, s = math.cos(ang), math.sin(ang)
    hx = math.sqrt((a * c) ** 2 + (b * s) ** 2)
    hy = math.sqrt((a * s) ** 2 + (b * c) ** 2)
    return [cx - hx, cy - hy, cx + hx, cy + hy]
