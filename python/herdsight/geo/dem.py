"""Terrain model for mission planning (Phase 5 plan, step 2; D-6).

A regular grid of ground heights (metres) in a local east-north frame: cell (row i, column j)
is at x = x0 + j * res, y = y0 + i * res. Sources in use: UK LIDAR (Environment Agency) and SRTM
or ALOS for South Africa, read into this grid. The planner flies a constant height above a
**smoothed** copy computed once at planning time (the drone flies level lines and does not chase
every bump), then checks the true clearance against the unsmoothed grid.

This is not `geo/terrain.py`, which fits a local ground plane to triangulated points for ranging.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class Dem:
    z: NDArray[np.float64]  # (rows, cols) ground height, metres
    x0: float
    y0: float
    res: float  # metres per cell

    @staticmethod
    def flat(width: float, height: float, res: float = 5.0, z: float = 0.0) -> Dem:
        rows, cols = int(np.ceil(height / res)) + 1, int(np.ceil(width / res)) + 1
        return Dem(np.full((rows, cols), float(z)), 0.0, 0.0, res)

    @staticmethod
    def from_function(width: float, height: float, res: float, f) -> Dem:  # noqa: ANN001
        ys = np.arange(int(np.ceil(height / res)) + 1) * res
        xs = np.arange(int(np.ceil(width / res)) + 1) * res
        xx, yy = np.meshgrid(xs, ys)
        return Dem(np.asarray(f(xx, yy), float), 0.0, 0.0, res)

    def height(self, pts: ArrayLike) -> NDArray[np.float64]:
        """Bilinear ground height at (x, y) points; points outside are clamped to the edge."""
        p = np.atleast_2d(np.asarray(pts, float))
        rows, cols = self.z.shape
        fx = np.clip((p[:, 0] - self.x0) / self.res, 0, cols - 1)
        fy = np.clip((p[:, 1] - self.y0) / self.res, 0, rows - 1)
        j0 = np.minimum(np.floor(fx).astype(int), cols - 2) if cols > 1 else np.zeros(len(p), int)
        i0 = np.minimum(np.floor(fy).astype(int), rows - 2) if rows > 1 else np.zeros(len(p), int)
        tx, ty = fx - j0, fy - i0
        j1, i1 = np.minimum(j0 + 1, cols - 1), np.minimum(i0 + 1, rows - 1)
        z = self.z
        return (
            z[i0, j0] * (1 - tx) * (1 - ty)
            + z[i0, j1] * tx * (1 - ty)
            + z[i1, j0] * (1 - tx) * ty
            + z[i1, j1] * tx * ty
        )

    def smoothed(self, sigma_m: float) -> Dem:
        """Gaussian-smoothed copy (sigma in metres)."""
        if sigma_m <= 0:
            return self
        return Dem(
            gaussian_filter(self.z, sigma_m / self.res, mode="nearest"), self.x0, self.y0, self.res
        )

    def with_error(self, sd_m: float, corr_m: float, rng: np.random.Generator) -> Dem:
        """A copy with spatially correlated height error (standard deviation sd_m, correlation
        length corr_m), the planner's imperfect copy of the terrain in simulation."""
        noise = gaussian_filter(rng.standard_normal(self.z.shape), corr_m / self.res, mode="wrap")
        noise *= sd_m / max(noise.std(), 1e-12)
        return Dem(self.z + noise, self.x0, self.y0, self.res)
