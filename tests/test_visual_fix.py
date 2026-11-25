"""Camera position fix by keyframe matching (herdsight.geo.visual_fix), on a synthetic map."""

import cv2
import numpy as np

from herdsight.geo.visual_fix import Keyframe, locate

M_PER_PX = 0.2


def textured_map(size=1200, seed=0):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.integers(0, 255, (size, size)).astype(np.uint8), (0, 0), 2)
    for _ in range(80):  # field-like edges and blobs give corners to match
        p1, p2 = rng.integers(0, size, 2), rng.integers(0, size, 2)
        cv2.line(
            img, tuple(int(v) for v in p1), tuple(int(v) for v in p2), int(rng.integers(0, 255)), 3
        )
    return img


def view(img, cx, cy, heading, out=(320, 240), scale=1.0):
    """Frame centred on map pixel (cx, cy), turned by heading (rad); returns frame, frame->map."""
    w, h = out
    rot = cv2.getRotationMatrix2D((cx, cy), np.degrees(heading), scale)
    rot[:, 2] += [w / 2 - cx, h / 2 - cy]
    frame = cv2.warpAffine(img, rot, (w, h))
    to_px = cv2.invertAffineTransform(rot)
    to_map = np.diag([M_PER_PX, -M_PER_PX]) @ to_px  # map pixels -> metres (y up)
    return frame, to_map


def test_finds_shifted_rotated_view_within_a_metre():
    m = textured_map()
    kf_img, kf_map = view(m, 600, 600, 0.0)
    kf = Keyframe(kf_img, kf_map)
    live, live_map = view(m, 640, 580, 0.6)  # 8 m east, 4 m north, turned 34 degrees
    truth = live_map @ np.array([160, 120, 1.0])
    fix = locate(live, [kf], prior_east_north=kf.centre + [30, 30])
    assert fix is not None and fix.inliers >= 40
    assert np.linalg.norm(fix.east_north - truth) < 1.0


def test_picks_the_right_keyframe_among_several():
    m = textured_map(seed=1)
    kfs = [Keyframe(*view(m, x, 600, 0.0)) for x in (300, 600, 900)]
    live, live_map = view(m, 900, 620, 1.2)
    fix = locate(live, kfs, prior_east_north=kfs[2].centre)
    assert fix is not None and fix.keyframe == 2
    assert np.linalg.norm(fix.east_north - live_map @ np.array([160, 120, 1.0])) < 1.0


def test_unrelated_ground_gives_no_fix():
    kf = Keyframe(*view(textured_map(seed=2), 600, 600, 0.0))
    live, _ = view(textured_map(seed=3), 600, 600, 0.0)
    assert locate(live, [kf], prior_east_north=kf.centre) is None


def test_blank_frame_gives_no_fix():
    kf = Keyframe(*view(textured_map(), 600, 600, 0.0))
    assert locate(np.full((240, 320), 128, np.uint8), [kf], kf.centre) is None


def test_fix_far_from_the_prior_is_refused():
    m = textured_map(seed=4)
    kf = Keyframe(*view(m, 600, 600, 0.0))
    live, _ = view(m, 620, 600, 0.3)
    far_prior = kf.centre + [80.0, 0.0]
    assert locate(live, [kf], far_prior) is not None  # no limit: the match is accepted
    assert locate(live, [kf], far_prior, max_jump_m=50.0) is None


def test_register_recovers_small_motion_between_frames():
    from herdsight.geo.visual_fix import register

    m = textured_map(seed=5)
    prev, prev_map = view(m, 600, 600, 0.10)
    cur, cur_map = view(m, 612, 596, 0.13)  # 2.4 m east, 0.8 m north, turned 1.7 degrees
    out = register(prev, cur)
    assert out is not None
    sim, inliers = out
    centre_in_prev = sim @ np.array([160, 120, 1.0])
    ground = prev_map @ np.array([*centre_in_prev, 1.0])
    assert np.linalg.norm(ground - cur_map @ np.array([160, 120, 1.0])) < 0.3


def test_register_refuses_unrelated_frames():
    from herdsight.geo.visual_fix import register

    a, _ = view(textured_map(seed=6), 600, 600, 0.0)
    b, _ = view(textured_map(seed=7), 600, 600, 0.0)
    assert register(a, b) is None
