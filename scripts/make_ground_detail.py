"""Make a fine-detail ground patch around the take-off point for the simulator. Simulation only.

Usage: python scripts/make_ground_detail.py [--size-m 120] [--cm-per-px 2.5] [--seed 0]

The IGN orthophoto draped over the simulated farm (scripts/fetch_ground_texture.py) has 20.6 cm
pixels. From 10-20 m up a camera sees 2-4 cm per pixel, so the orthophoto is only blur there,
while real pasture shows grass tufts, soil, dung and wheel marks. This patch keeps the
orthophoto's layout (the centre --size-m square, upsampled) and adds static detail: noise at
scales from 25 cm to 2 m (finer grain flickers from frame to frame when seen from 83 m), with
contrast of about +/-12%, fixed by --seed. It is an assumption of the simulation, stated wherever
a result depends on it: the ground near the landing point has visible texture, as grass and soil
do. Writes $HERDSIGHT_STORAGE/datasets/raw/ign_bdortho/ground_detail_<size>m_seed<seed>.jpg.
"""

from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from herdsight.utils.paths import storage_path

SOURCE = "charolles_pasture_z19"
SCALES_M = (0.25, 0.5, 1.0, 2.0)  # finer grain flickers (aliases) when seen from survey height
CONTRAST = 0.12


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size-m", type=float, default=120.0)
    ap.add_argument("--cm-per-px", type=float, default=2.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--scales", type=float, nargs="+", default=list(SCALES_M), help="grain sizes (m)"
    )
    ap.add_argument("--name", default=None, help="output file name")
    args = ap.parse_args(argv)
    folder = storage_path("raw", "ign_bdortho")
    meta = json.loads((folder / f"{SOURCE}.json").read_text())
    ortho = cv2.imread(str(folder / f"{SOURCE}.jpg"))
    m_per_px = float(meta["metres_per_pixel"])
    h, w = ortho.shape[:2]
    half = args.size_m / 2 / m_per_px
    crop = ortho[int(h / 2 - half) : int(h / 2 + half), int(w / 2 - half) : int(w / 2 + half)]
    out_px = int(round(args.size_m * 100 / args.cm_per_px))
    big = cv2.resize(crop, (out_px, out_px), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    rng = np.random.default_rng(args.seed)
    detail = np.zeros((out_px, out_px), np.float32)
    for s in args.scales:
        cells = max(2, int(round(args.size_m / s)))
        layer = rng.normal(0, 1, (cells, cells)).astype(np.float32)
        detail += cv2.resize(layer, (out_px, out_px), interpolation=cv2.INTER_CUBIC)
    detail /= detail.std()
    big *= (1 + CONTRAST * np.clip(detail, -2, 2) / 2)[..., None]  # +/-12% at two s.d.
    name = args.name or f"ground_detail_{int(args.size_m)}m_seed{args.seed}.jpg"
    cv2.imwrite(
        str(folder / name), np.clip(big, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 90]
    )
    print(f"wrote {folder / name}: {out_px} px, {args.cm_per_px} cm per pixel, {args.size_m} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
