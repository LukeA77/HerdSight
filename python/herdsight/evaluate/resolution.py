"""Image degradation and the pass rule for the resolution study (Phase 2, step G11).

An image is reduced with area resampling, then given blur, sensor noise and JPEG compression, so
that it resembles a photo taken from higher up. The result is a lower bound on the pixels an
animal needs: it is measured on adult cattle, and calves are not in the data.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter


def image_seed(key: str, scale: float, seed: int) -> int:
    """A seed that depends only on the image name, the scale and the study seed."""
    digest = hashlib.sha256(f"{key}|{scale:.4f}|{seed}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def degrade(
    image: Image.Image,
    scale: float,
    blur_sigma_px: float,
    noise_sd: float,
    jpeg_quality: int,
    seed: int,
) -> Image.Image:
    """The degraded RGB image. Its size is the input size times `scale`, rounded."""
    if not 0 < scale <= 1:
        raise ValueError("scale must be in (0, 1]")
    rgb = image.convert("RGB")
    size = (max(round(rgb.width * scale), 1), max(round(rgb.height * scale), 1))
    out = rgb if size == rgb.size else rgb.resize(size, Image.Resampling.BOX)
    if blur_sigma_px > 0:
        out = out.filter(ImageFilter.GaussianBlur(blur_sigma_px))
    if noise_sd > 0:
        rng = np.random.default_rng(seed)
        arr = np.asarray(out, np.float32)
        arr = arr + rng.normal(0.0, noise_sd, arr.shape)
        out = Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8))
    buffer = io.BytesIO()
    out.save(buffer, "JPEG", quality=jpeg_quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


@dataclass(frozen=True)
class ScaleResult:
    """Change in recall against native resolution at one scale."""

    scale: float
    recall_change: float
    ci_low: float


def smallest_passing(
    results: Sequence[ScaleResult], max_loss: float
) -> tuple[float | None, float | None]:
    """Smallest scale that passes, walking down from the largest and stopping at the first failure.

    Returns (by the point estimate, by the lower end of the 95% interval). A scale below a failed
    one never counts, so a lucky result at a small size cannot pass.
    """
    point: float | None = None
    cautious: float | None = None
    point_open = cautious_open = True
    for r in sorted(results, key=lambda x: -x.scale):
        point_open = point_open and r.recall_change >= -max_loss
        cautious_open = cautious_open and r.ci_low >= -max_loss
        if point_open:
            point = r.scale
        if cautious_open:
            cautious = r.scale
    return point, cautious
