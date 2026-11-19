import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from herdsight.geo.register import apply_homography, register_pair  # noqa: E402


def _texture(h=900, w=1200, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8)
    img = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    for _ in range(300):  # blobs so SIFT has distinctive structure
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.circle(
            img, c, int(rng.integers(3, 25)), tuple(int(x) for x in rng.integers(0, 255, 3)), -1
        )
    return img


def test_recovers_a_known_shift_and_rotation():
    world = _texture(1400, 1800)
    # B is a crop of the world rotated by 7 degrees and shifted; A is a plain crop.
    a = world[200:1100, 200:1400].copy()
    M = cv2.getRotationMatrix2D((900, 700), 7.0, 1.0)
    rotated = cv2.warpAffine(world, M, (1800, 1400))
    b = rotated[350:1250, 450:1650].copy()
    reg = register_pair(a, b, work_px=1200)
    assert reg.accepted, reg.reason
    # Ground truth mapping of B pixels to A pixels
    Minv = cv2.invertAffineTransform(M)
    pts_b = np.array([[100.0, 100.0], [600.0, 450.0], [1100.0, 800.0]])
    world_pts = np.column_stack([pts_b + [450, 350], np.ones(3)]) @ Minv.T
    expect_a = world_pts - [200, 200]
    np.testing.assert_allclose(apply_homography(reg.H_ab, pts_b), expect_a, atol=2.0)
    assert reg.reprojection_px < 2.0


def test_unrelated_images_are_rejected():
    reg = register_pair(_texture(seed=1), _texture(seed=2))
    assert not reg.accepted


def test_geotag_disagreement_rejects():
    world = _texture(1400, 1800)
    a, b = world[200:1100, 200:1400].copy(), world[300:1200, 400:1600].copy()
    ok = register_pair(a, b, prior_b_centre_in_a=[800, 550], max_disagreement_px=50)
    assert ok.accepted
    bad = register_pair(a, b, prior_b_centre_in_a=[100, 100], max_disagreement_px=50)
    assert not bad.accepted and bad.reason == "disagrees with geotag"
