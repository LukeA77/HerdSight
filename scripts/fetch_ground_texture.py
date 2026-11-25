"""Fetch a real aerial image of pasture to texture the simulator's ground (Phase 5, GPS-loss work).

Usage: python scripts/fetch_ground_texture.py [--lat 46.4245 --lon 4.2884 --size-m 1200 --zoom 19]

The simulated ground was one flat colour, so a downward camera had nothing to match, which made
camera-based navigation untestable. This stitches IGN orthophoto tiles (BD ORTHO, 20 cm class)
from the public Geoplateforme WMTS service (data.geopf.fr/wmts, layer ORTHOIMAGERY.ORTHOPHOTOS,
Web Mercator tile set PM) into one square image centred on the given point, and writes it with a
JSON record of source, licence, attribution, extent, metres per pixel and SHA-256 to
$HERDSIGHT_STORAGE/datasets/raw/ign_bdortho/. The default point is hedged pasture near Charolles
(Saone-et-Loire, France), cattle country with fields, hedges, tracks and woodland.
Licence: Etalab Open Licence 2.0 (Licence Ouverte), which allows commercial reuse with attribution:
"Source: IGN - BD ORTHO". Imagery only; no animals in it are used.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import time
import urllib.request

from PIL import Image

from herdsight.utils.paths import storage_path

URL = (
    "https://data.geopf.fr/wmts?SERVICE=WMTS&REQUEST=GetTile&VERSION=1.0.0"
    "&LAYER=ORTHOIMAGERY.ORTHOPHOTOS&STYLE=normal&TILEMATRIXSET=PM"
    "&TILEMATRIX={z}&TILEROW={row}&TILECOL={col}&FORMAT=image/jpeg"
)
TILE = 256
EARTH = 156543.03392804097  # metres per pixel at zoom 0 at the equator (Web Mercator)


def global_px(lat: float, lon: float, z: int) -> tuple[float, float]:
    n = TILE * 2**z
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=46.4245)
    ap.add_argument("--lon", type=float, default=4.2884)
    ap.add_argument("--size-m", type=float, default=1200.0)
    ap.add_argument("--zoom", type=int, default=19)
    ap.add_argument("--name", default="charolles_pasture")
    args = ap.parse_args(argv)
    m_per_px = EARTH * math.cos(math.radians(args.lat)) / 2**args.zoom
    half = int(round(args.size_m / m_per_px / 2))
    cx, cy = global_px(args.lat, args.lon, args.zoom)
    x0, y0, x1, y1 = int(cx) - half, int(cy) - half, int(cx) + half, int(cy) + half
    out = Image.new("RGB", (x1 - x0, y1 - y0))
    tiles = 0
    for row in range(y0 // TILE, (y1 - 1) // TILE + 1):
        for col in range(x0 // TILE, (x1 - 1) // TILE + 1):
            url = URL.format(z=args.zoom, row=row, col=col)
            req = urllib.request.Request(url, headers={"User-Agent": "sim-texture-fetch/1.0"})
            for attempt in range(4):
                try:
                    data = urllib.request.urlopen(req, timeout=30).read()
                    break
                except OSError:
                    if attempt == 3:
                        raise
                    time.sleep(2 + 3 * attempt)
            im = Image.open(io.BytesIO(data)).convert("RGB")
            out.paste(im, (col * TILE - x0, row * TILE - y0))
            tiles += 1
            time.sleep(0.05)  # be gentle with a free public service
    dest_dir = storage_path("raw", "ign_bdortho", create=True)
    img_path = dest_dir / f"{args.name}_z{args.zoom}.jpg"
    out.save(img_path, quality=92)
    record = {
        "source": "IGN BD ORTHO, Geoplateforme WMTS data.geopf.fr/wmts, ORTHOIMAGERY.ORTHOPHOTOS",
        "licence": "Etalab Open Licence 2.0; commercial reuse allowed with attribution",
        "attribution": "Source: IGN - BD ORTHO",
        "centre_lat_lon": [args.lat, args.lon],
        "zoom": args.zoom,
        "metres_per_pixel": m_per_px,
        "size_px": list(out.size),
        "size_m": [out.size[0] * m_per_px, out.size[1] * m_per_px],
        "tiles_fetched": tiles,
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "image": img_path.name,
        "image_sha256": hashlib.sha256(img_path.read_bytes()).hexdigest(),
    }
    (dest_dir / f"{args.name}_z{args.zoom}.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
