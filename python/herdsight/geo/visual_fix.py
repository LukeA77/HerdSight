"""Position fix from the downward camera by matching to the survey's own keyframes.

For the GPS-loss return home (_drafts/gps-loss-camera-return-spec.md): during the survey each
still is kept as a keyframe with the transform from its pixels to map coordinates (metres east and
north of an origin). After GPS is lost, a live frame is matched to the keyframes nearest a prior
position (the last estimate, which may have drifted). ORB features (Rublee et al., ICCV 2011) are
matched with Lowe's ratio test, and a similarity transform (rotation, scale, shift) is fitted with
RANSAC (cv2.estimateAffinePartial2D). A fix is the live frame's centre carried into map
coordinates through the best keyframe; it is given only with at least `min_inliers` consistent
matches, otherwise None, so the caller can fall back rather than fly on a guess.

Both images are first evened out with CLAHE (contrast-limited adaptive histogram equalisation,
Zuiderveld 1994), which makes the features less sensitive to a change of light between the survey
and the return. A fix further than `max_jump_m` from the prior is refused: the prior's error is
bounded by dead reckoning, so a match that far away is a mismatch, not a position.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Keyframe:
    image: np.ndarray  # greyscale uint8
    to_map: np.ndarray  # 2x3 affine: keyframe pixel (x, y, 1) -> map metres (east, north)
    centre: np.ndarray = field(init=False)
    keypoints: tuple = field(init=False, repr=False)
    descriptors: np.ndarray | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        h, w = self.image.shape[:2]
        self.centre = self.to_map @ np.array([w / 2, h / 2, 1.0])
        self.keypoints, self.descriptors = _orb().detectAndCompute(_even(self.image), None)


@dataclass(frozen=True)
class Fix:
    east_north: np.ndarray
    inliers: int
    keyframe: int


def _orb(n: int = 2000) -> cv2.ORB:
    return cv2.ORB_create(nfeatures=n)


def _even(image: np.ndarray) -> np.ndarray:
    """Local contrast evening (CLAHE, clip limit 2, 8 x 8 tiles) before feature detection."""
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(image)


def locate(
    live: np.ndarray,
    keyframes: list[Keyframe],
    prior_east_north: np.ndarray,
    nearest: int = 6,
    ratio: float = 0.8,
    min_inliers: int = 40,
    max_jump_m: float | None = None,
) -> Fix | None:
    """Map position of the live frame's centre, or None if no keyframe matches well enough.

    max_jump_m: refuse a fix further than this from the prior (None: no limit).
    """
    if not keyframes:
        return None
    kp, des = _orb().detectAndCompute(_even(live), None)
    if des is None or len(kp) < min_inliers:
        return None
    d = [float(np.linalg.norm(k.centre - prior_east_north)) for k in keyframes]
    order = np.argsort(d)[:nearest]
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    h, w = live.shape[:2]
    best: Fix | None = None
    for i in order:
        k = keyframes[int(i)]
        if k.descriptors is None or len(k.keypoints) < 2:
            continue
        pairs = matcher.knnMatch(des, k.descriptors, k=2)
        good = [m for m, *rest in pairs if rest and m.distance < ratio * rest[0].distance]
        if len(good) < min_inliers:
            continue
        src = np.float32([kp[m.queryIdx].pt for m in good])
        dst = np.float32([k.keypoints[m.trainIdx].pt for m in good])
        sim, mask = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0
        )
        if sim is None:
            continue
        n_in = int(mask.sum())
        if n_in < min_inliers or (best is not None and n_in <= best.inliers):
            continue
        centre_kf = sim @ np.array([w / 2, h / 2, 1.0])
        where = k.to_map @ np.array([*centre_kf, 1.0])
        if max_jump_m is not None and np.linalg.norm(where - prior_east_north) > max_jump_m:
            continue
        best = Fix(where, n_in, int(i))
    return best


def register(
    previous: np.ndarray, current: np.ndarray, ratio: float = 0.8, min_inliers: int = 30
) -> tuple[np.ndarray, int] | None:
    """Similarity transform (2x3) taking current-frame pixels to previous-frame pixels, or None.

    Frame-to-frame visual odometry: consecutive frames share light and nearly all their ground, so
    this succeeds far more often than matching to a keyframe from the survey. Chained, it gives the
    drone's movement between keyframe fixes; its drift is reset by each fix.
    """
    orb = _orb()
    kp1, des1 = orb.detectAndCompute(_even(current), None)
    kp0, des0 = orb.detectAndCompute(_even(previous), None)
    if des1 is None or des0 is None or len(kp1) < min_inliers or len(kp0) < 2:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des1, des0, k=2)
    good = [m for m, *rest in pairs if rest and m.distance < ratio * rest[0].distance]
    if len(good) < min_inliers:
        return None
    src = np.float32([kp1[m.queryIdx].pt for m in good])
    dst = np.float32([kp0[m.trainIdx].pt for m in good])
    sim, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if sim is None or int(mask.sum()) < min_inliers:
        return None
    return sim, int(mask.sum())
