"""Register overlapping nadir stills from their content (D-27).

Pasture is close to flat at survey scale, so overlapping nadir stills are related by a
homography. SIFT features (OpenCV; the SIFT patent has expired) are matched with a ratio test
and the homography is fitted by RANSAC, which ignores matches on moving animals.

Features are found on a reduced copy of each image (`work_px` on the long side) for speed. This
is registration only: detection and counting stay at full resolution (D-8). The homography is
returned in full-resolution pixels.

A registration is accepted only if it has enough RANSAC inliers and agrees with the geotag
prediction of where image B's centre falls in image A (`max_disagreement_px`); otherwise the
caller falls back to the geotag alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class Registration:
    H_ab: NDArray[np.float64] | None  # maps image-B pixels to image-A pixels (3x3), or None
    inliers: int
    matches: int
    reprojection_px: float | None  # median inlier error, full-resolution pixels
    accepted: bool
    reason: str


def _gray_small(img: NDArray[np.uint8], work_px: int) -> tuple[NDArray[np.uint8], float]:
    import cv2

    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    s = work_px / max(g.shape[:2])
    if s < 1:
        g = cv2.resize(
            g, (round(g.shape[1] * s), round(g.shape[0] * s)), interpolation=cv2.INTER_AREA
        )
    else:
        s = 1.0
    return g, s


def apply_homography(H: ArrayLike, xy: ArrayLike) -> NDArray[np.float64]:
    """Map points (n, 2) through a 3x3 homography."""
    H = np.asarray(H, float)
    p = np.asarray(xy, float).reshape(-1, 2)
    q = np.column_stack([p, np.ones(len(p))]) @ H.T
    return q[:, :2] / q[:, 2:3]


@dataclass(frozen=True)
class Features:
    xy: NDArray[np.float32]  # keypoint positions, full-resolution pixels
    desc: NDArray[np.float32] | None
    width: int
    height: int


def features(img: NDArray[np.uint8], work_px: int = 1600, n: int = 8000) -> Features:
    """SIFT keypoints (in full-resolution pixels) and descriptors of one image."""
    import cv2

    g, s = _gray_small(img, work_px)
    kp, desc = cv2.SIFT_create(nfeatures=n).detectAndCompute(g, None)
    xy = np.float32([k.pt for k in kp]).reshape(-1, 2) / s
    return Features(xy, desc, img.shape[1], img.shape[0])


def register_features(
    fa: Features,
    fb: Features,
    min_inliers: int = 40,
    ratio: float = 0.75,
    ransac_px: float = 12.0,
    prior_b_centre_in_a: ArrayLike | None = None,
    max_disagreement_px: float | None = None,
) -> Registration:
    """Homography from image B to image A from precomputed features, with acceptance checks.
    `ransac_px` is in full-resolution pixels."""
    import cv2

    if fa.desc is None or fb.desc is None or len(fa.xy) < 8 or len(fb.xy) < 8:
        return Registration(None, 0, 0, None, False, "too few features")
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(fb.desc, fa.desc, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < ratio * n.distance]
    if len(good) < 8:
        return Registration(None, 0, len(good), None, False, "too few matches")
    pb = fb.xy[[m.queryIdx for m in good]]
    pa = fa.xy[[m.trainIdx for m in good]]
    H, mask = cv2.findHomography(pb, pa, cv2.RANSAC, ransac_px)
    if H is None:
        return Registration(None, 0, len(good), None, False, "no homography")
    inl = mask.ravel().astype(bool)
    err = np.linalg.norm(apply_homography(H, pb[inl]) - pa[inl], axis=1)
    reg = dict(
        H_ab=H, inliers=int(inl.sum()), matches=len(good), reprojection_px=float(np.median(err))
    )
    if inl.sum() < min_inliers:
        return Registration(**reg, accepted=False, reason="too few inliers")
    if prior_b_centre_in_a is not None and max_disagreement_px is not None:
        c = apply_homography(H, [[fb.width / 2, fb.height / 2]])[0]
        if np.linalg.norm(c - np.asarray(prior_b_centre_in_a, float)) > max_disagreement_px:
            return Registration(**reg, accepted=False, reason="disagrees with geotag")
    return Registration(**reg, accepted=True, reason="ok")


def register_pair(
    img_a: NDArray[np.uint8],
    img_b: NDArray[np.uint8],
    work_px: int = 1600,
    **kw: object,
) -> Registration:
    """Homography from image B to image A (see register_features for the checks)."""
    return register_features(features(img_a, work_px), features(img_b, work_px), **kw)  # type: ignore[arg-type]
