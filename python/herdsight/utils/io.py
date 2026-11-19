"""Reading images from an external drive that may drop out briefly.

USB drives can disappear for a few seconds (Windows logs disk event 153 and the read fails with
"Invalid argument"). Long runs retry a failed read rather than stopping; each retry is printed,
so dropouts are visible in the run log.
"""

from __future__ import annotations

import time
from pathlib import Path

from PIL import Image


def load_rgb(path: Path | str, retries: int = 6, wait_s: float = 10.0) -> Image.Image:
    """Open an image fully into memory as RGB, retrying on OSError."""
    for attempt in range(retries + 1):
        try:
            with Image.open(path) as im:
                return im.convert("RGB")
        except OSError as e:
            if attempt == retries:
                raise
            print(f"read failed ({e}); retrying {path} in {wait_s:.0f} s", flush=True)
            time.sleep(wait_s)
    raise AssertionError("unreachable")
