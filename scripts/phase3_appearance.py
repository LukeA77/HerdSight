"""Phase 3 step 4 (H25): two appearance descriptors for every detection of the pair tables.

Usage (in an environment with PyTorch and transformers, e.g. the zero-shot one on the storage
drive; the model is cached under HF_HOME):
  HF_HOME=$HERDSIGHT_STORAGE/cache/huggingface PYTHONPATH=python \
    <env python> scripts/phase3_appearance.py

For each source (labels, dfine) and each detection saved by scripts/phase3_pairs.py, a square crop
centred on the box, 1.25 times its long side (at least 32 pixels), is described two ways
(configs/phase3.yaml, appearance_descriptors):

* colour_texture: HSV colour histogram (16 x 4 x 4 bins) and a rotation-invariant uniform local
  binary pattern histogram (radius 2, 16 points: 18 bins), each normalised to sum 1 and
  concatenated; compared with the chi-squared distance. Nadir views have no "up", so the texture
  part is rotation invariant and colour histograms are by nature.
* pretrained: the class-token embedding of DINOv2 ViT-S/14 (facebook/dinov2-small) for the crop
  resized to 224 pixels, L2-normalised; compared with the cosine distance. CPU only.

Writes $HERDSIGHT_STORAGE/processed/phase3/appearance_<source>.npz: per flight, per still, the
descriptors of its detections in the saved order.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from herdsight.data.icaerus import default_root
from herdsight.utils.paths import storage_path

MODEL = "facebook/dinov2-small"
SOURCES = ("labels", "dfine")
P, R = 16, 2.0
HSV_BINS = (16, 4, 4)


def lbp_riu2(gray: np.ndarray) -> np.ndarray:
    """Rotation-invariant uniform LBP histogram (P + 2 bins) with bilinear sampling."""
    h, w = gray.shape
    yy, xx = np.mgrid[R : h - R, R : w - R].astype(float) if h > 2 * R + 1 else (None, None)
    if yy is None or yy.size == 0:
        return np.full(P + 2, 1.0 / (P + 2))
    centre = gray[int(R) : h - int(R), int(R) : w - int(R)][: yy.shape[0], : yy.shape[1]]
    bits = []
    for k in range(P):
        a = 2 * np.pi * k / P
        y, x = yy - R * np.sin(a), xx + R * np.cos(a)
        y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
        y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
        fy, fx = y - y0, x - x0
        v = (
            gray[y0, x0] * (1 - fy) * (1 - fx)
            + gray[y0, x1] * (1 - fy) * fx
            + gray[y1, x0] * fy * (1 - fx)
            + gray[y1, x1] * fy * fx
        )
        bits.append(v >= centre)
    b = np.stack(bits).astype(int)
    transitions = np.abs(np.diff(np.concatenate([b, b[:1]]), axis=0)).sum(axis=0)
    code = np.where(transitions <= 2, b.sum(axis=0), P + 1)
    hist = np.bincount(code.ravel(), minlength=P + 2).astype(float)
    return hist / max(hist.sum(), 1.0)


def colour_texture(crop: Image.Image) -> np.ndarray:
    hsv = np.asarray(crop.convert("HSV"), float) / 256.0
    idx = [np.minimum((hsv[..., c] * n).astype(int), n - 1) for c, n in enumerate(HSV_BINS)]
    flat = (idx[0] * HSV_BINS[1] + idx[1]) * HSV_BINS[2] + idx[2]
    hist = np.bincount(flat.ravel(), minlength=int(np.prod(HSV_BINS))).astype(float)
    hist /= max(hist.sum(), 1.0)
    gray = np.asarray(crop.convert("L"), float)
    return np.concatenate([hist, lbp_riu2(gray)])


def square_crop(img: Image.Image, box: list[float]) -> Image.Image:
    x0, y0, x1, y1 = box
    side = max(1.25 * max(x1 - x0, y1 - y0), 32.0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return img.crop(
        (round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2))
    )


def main() -> int:
    torch.set_num_threads(max(1, (torch.get_num_threads() or 2) - 1))
    proc = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).eval()
    root = default_root()
    d3 = storage_path("processed", "phase3")
    for src in SOURCES:
        dets = json.loads(str(np.load(d3 / f"pairs_{src}.npz")["detections"]))
        out: dict[str, list] = {}
        t0, n_done = time.time(), 0
        for key, rec in sorted(dets.items()):
            farm, flight = key.split("/")
            per_still = []
            for name, boxes in zip(rec["images"], rec["boxes"], strict=True):
                if not boxes:
                    per_still.append({"colour_texture": [], "pretrained": []})
                    continue
                img = Image.open(root / farm / "JPGImages" / flight / name).convert("RGB")
                crops = [square_crop(img, b) for b in boxes]
                ct = [colour_texture(c).tolist() for c in crops]
                with torch.no_grad():
                    emb = []
                    for k in range(0, len(crops), 32):
                        x = proc(images=crops[k : k + 32], return_tensors="pt")
                        e = model(**x).last_hidden_state[:, 0]
                        emb.append(torch.nn.functional.normalize(e, dim=1).numpy())
                per_still.append(
                    {"colour_texture": ct, "pretrained": np.concatenate(emb).round(5).tolist()}
                )
                n_done += len(crops)
            out[key] = per_still
            print(f"{src} {key}: {n_done} crops, {time.time() - t0:.0f} s", flush=True)
        np.savez_compressed(d3 / f"appearance_{src}.npz", descriptors=np.array(json.dumps(out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
