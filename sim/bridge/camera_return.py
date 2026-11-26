"""Camera return home after GPS loss: teach during the survey, repeat on the way back. Simulation.

Usage (inside the simulation image, started by run_flight.sh with VISION=1):
  python3 camera_return.py --plan plan.yaml --log camera_return.csv [--validate]

Design: _drafts/gps-loss-camera-return-spec.md. The camera is fixed to the drone's body, pointing
down (sim/bridge/add_body_camera.py), so it tilts with the drone as a camera without a gimbal
would. Each frame is tied to PX4's state at the frame's own simulation time (the image header),
taken from a short history of PX4's position, velocity and attitude: the point on the ground at
the image centre is the drone's position plus the height times the tilt of the optical axis.

Teach: while GPS is good, a frame becomes a keyframe, with the map position of its centre and its
heading, whenever the drone is 40% of a footprint from every kept keyframe at a similar height,
including during the climb over the take-off point so that the descent home can be matched too.
Repeat: once GPS is lost (GPS reports no 3D fix for 1.5 s, or falls silent for 10 s), each new
frame is matched to keyframes
taken at a similar height near the last position (herdsight.geo.visual_fix.locate, with the 75 m
jump limit of the third camera test); the drone's position is the matched image centre minus the
tilt offset. EKF2 only starts vision fusion when samples arrive at least every 0.2 s, so the camera
runs at 5 frames per second and every frame is tried; a fix is carried forward for at most 1 s
(with the distance PX4's velocity estimate has covered since its frame) only to bridge a missed
frame, and sent at 10 Hz to /fmu/in/vehicle_visual_odometry (horizontal position; EKF2_EV_CTRL
bit 0). Carrying fixes for several seconds fed PX4's own drift back into it, so it was dropped.
With no new fix for 1 s the node stops sending and PX4 falls back on its own. After the first
fix the node asks PX4 to return to launch.

--validate also tries a fix on every fourth frame while GPS is good, against keyframes at least
20 s old, and logs it as a "check" (never sent to PX4), to measure fix accuracy before any loss.
Every logged row carries Gazebo's true drone position for scoring; the node never navigates on it.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import sys
import threading
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
import yaml
from gz.msgs10.clock_pb2 import Clock
from gz.msgs10.image_pb2 import Image as ImgMsg
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node as GzNode
from PIL import Image
from px4_msgs.msg import (
    SensorGps,
    VehicleAttitude,
    VehicleCommand,
    VehicleGlobalPosition,
    VehicleLocalPosition,
    VehicleOdometry,
)
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from add_body_camera import HEIGHT_PX, TOPIC, WIDTH_PX  # noqa: E402

from herdsight.geo.visual_fix import Keyframe, locate, register  # noqa: E402

# GPS loss: PX4 keeps publishing GPS messages that report no 3D fix for this long, or no GPS
# message arrives at all for GPS_SILENT_S. Gaps between messages alone are not loss: simulator
# stalls at start-up made 3 s gaps and false returns during the teach square (test 5, 25 Nov)
GPS_NOFIX_S = 1.5
GPS_SILENT_S = 10.0
NEW_KEYFRAME_SHARE = 0.4  # of the footprint width
SAME_HEIGHT_SHARE = 0.15
MIN_KEYFRAME_HEIGHT_M = 4.0
MATCH_HEIGHT_RATIO = 2.0  # only keyframes from half to twice the current height
MAX_JUMP_M = 75.0
FIX_SIGMA_M = 1.0
# a fix is used only if it agrees with the chained position (within this many metres, or 3 sigma),
# or implies the same correction to the chain as the rejected fix before it: single wrong matches
# are dropped, but a drifted chain is still corrected by two fixes that agree (test 3 compared raw
# positions, which differ by the drone's own movement, so a drifted chain was never corrected)
FIX_AGREE_M = 6.0
FIX_MIN_INLIERS = 150  # and at least this share of the ratio-test matches must be RANSAC inliers:
FIX_MIN_SHARE = 0.5
QUALITY_ABOVE_M = 20.0  # the quality limits apply above this height only: the offline study found
# no wrong fixes below 20 m, and close-up frames near the ground have fewer matches (test 4)
# Final descent: below this height, after this long with no accepted fix, stop sending positions so
# PX4 lands where it is; a frozen position sent near the ground made PX4 fly off (test 4)
LAND_IN_PLACE_BELOW_M = 10.0
NO_FIX_LAND_S = 3.0
FIX_PAIR_M = 3.0
CARRY_SIGMA_M_S = 0.3  # added uncertainty per second of carrying a fix forward
CARRY_MAX_S = 1.0  # only bridges a missed frame; longer carrying fed drift back into PX4
GROUND_HEIGHT_M = 0.5  # below this for GROUND_HOLD_S, the drone is on the ground
GROUND_HOLD_S = 3.0
VO_SIGMA_STEP_M = 0.15  # uncertainty added per chained frame-to-frame step
VO_SIGMA_MISS_M = 0.5  # added when a step falls back on PX4's velocity
VO_MAX_SPEED_M_S = 20.0  # fastest plausible ground speed (cruise plus a strong tailwind)
VO_STEP_SLACK_M = 2.0
PUBLISH_PERIOD_US = 100_000
HISTORY_S = 10.0
CHECK_EVERY = 4
CHECK_MIN_AGE_US = 20_000_000


def rotation(q: np.ndarray) -> np.ndarray:
    """Body (FRD) to local NED rotation matrix from PX4's quaternion (w, x, y, z)."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def frame_to_map(east: float, north: float, slant: float, theta: float, fov: float) -> np.ndarray:
    """2x3 affine: frame pixel -> metres (east, north). theta: image-top direction, ENU yaw;
    (east, north): ground point at the image centre; slant: distance along the optical axis."""
    s = 2 * slant * math.tan(fov / 2) / WIDTH_PX
    up = np.array([math.cos(theta), math.sin(theta)])
    right = np.array([math.sin(theta), -math.cos(theta)])
    a = np.column_stack([right * s, -up * s])
    c = np.array([east, north]) - a @ np.array([WIDTH_PX / 2, HEIGHT_PX / 2])
    return np.column_stack([a, c])


class State:
    """PX4's state at one instant: position (NED), height above home, attitude, distance moved."""

    def __init__(
        self,
        t_us: int,
        ned: np.ndarray,
        q: np.ndarray,
        moved: np.ndarray,
        height_agl: float | None = None,
    ) -> None:
        self.t_us, self.ned, self.q, self.moved = t_us, ned, q, moved
        r = rotation(q)
        axis = r @ np.array([0.0, 0.0, 1.0])  # optical axis (body down) in NED
        # height above the ground: the rangefinder's when there is one (without GPS, PX4's own
        # height drifts several metres during the descent: development flight, 25 Nov)
        self.height = float(-ned[2]) if height_agl is None else float(height_agl)
        self.cos_tilt = max(float(axis[2]), 0.2)
        # ground offset of the image centre from the point under the drone (east, north)
        self.offset_en = self.height * np.array([axis[1], axis[0]]) / self.cos_tilt
        fwd = r @ np.array([1.0, 0.0, 0.0])
        self.theta = math.atan2(fwd[0], fwd[1])  # ENU yaw of the image top (body front)
        self.slant = self.height / self.cos_tilt

    @property
    def en(self) -> np.ndarray:
        return np.array([self.ned[1], self.ned[0]])


class CameraReturn(Node):
    def __init__(
        self,
        plan: dict,
        log_path: Path,
        world: str,
        validate: bool,
        lag_s: float = 0.0,
        record: Path | None = None,
        rangefinder: bool = False,
        return_height: float = 0.0,
    ) -> None:
        super().__init__("camera_return")
        self.fov = 2 * math.atan(float(plan["footprint_m"][0]) / 2 / float(plan["height_agl_m"]))
        self.validate = validate
        self.rangefinder = rangefinder
        self.return_height = return_height
        self.descending = False
        self.global_alt: float | None = None
        self.lock = threading.Lock()
        self.history: deque[State] = deque()
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.moved = np.zeros(2)
        self.prev_est_us: int | None = None
        self.frame: tuple[int, np.ndarray] | None = None  # image time (us), grey image
        self.used_frame_us = 0
        self.frames_seen = 0
        self.truth: deque[tuple[int, float, float]] = deque(maxlen=400)  # sim us, east, north
        self.gps_us = 0
        self.gps_any_us = 0
        self.nofix_since_us: int | None = None
        self.sim_us = 0  # latest Gazebo simulation time
        self.att_msgs = 0
        self.ground_since: int | None = None
        self.disarmed = False
        self.lag_us = int(lag_s * 1e6)
        self.record = record
        if record is not None:
            record.mkdir(parents=True, exist_ok=True)
        self.px4_us = 0  # latest PX4 time (as converted by the DDS bridge)
        self.keyframes: list[Keyframe] = []
        self.kf_info: list[tuple[float, float, float, int]] = []  # centre e, n, height, time us
        self.lost = False
        self.last_fix: tuple[int, np.ndarray] | None = None
        self.anchor: tuple[np.ndarray, np.ndarray, int, float] | None = (
            None  # e/n, moved, us, sigma
        )
        self.vo_pos: np.ndarray | None = None  # chained camera position (east, north)
        self.vo_sigma = FIX_SIGMA_M
        self.pending: np.ndarray | None = None
        self.vision_stopped = False
        self.prev: tuple[np.ndarray, State, np.ndarray] | None = None  # grey, state, position
        self.last_pub_us = 0
        self.rtl_sent = False
        self.log = open(log_path, "w", newline="")
        self.out = csv.writer(self.log)
        self.out.writerow(
            ["t_s", "event", "est_e", "est_n", "fix_e", "fix_n", "true_e", "true_n", "height",
             "inliers", "keyframes", "tilt_off_e", "tilt_off_n", "att_msgs"]
        )  # fmt: skip
        qos = qos_profile_sensor_data
        self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1", self.on_est, qos
        )
        self.create_subscription(VehicleAttitude, "/fmu/out/vehicle_attitude", self.on_att, qos)
        self.create_subscription(SensorGps, "/fmu/out/vehicle_gps_position", self.on_gps, qos)
        self.create_subscription(
            VehicleGlobalPosition, "/fmu/out/vehicle_global_position", self.on_global, qos
        )
        self.ev_pub = self.create_publisher(VehicleOdometry, "/fmu/in/vehicle_visual_odometry", qos)
        self.cmd_pub = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", qos)
        self.gz = GzNode()
        self.gz.subscribe(ImgMsg, f"/{TOPIC}", self.on_image)
        self.gz.subscribe(Pose_V, f"/world/{world}/pose/info", self.on_poses)
        self.gz.subscribe(Clock, f"/world/{world}/clock", self.on_clock)
        # image matching runs in its own thread, so that sending positions (from on_est) is never
        # held up by it: EKF2 drops vision fusion after a 0.2 s gap
        self.create_timer(0.1, self.step, callback_group=MutuallyExclusiveCallbackGroup())

    # inputs --------------------------------------------------------------------------------
    def on_att(self, m: VehicleAttitude) -> None:
        self.q = np.array(m.q, float)
        self.att_msgs += 1

    def on_est(self, m: VehicleLocalPosition) -> None:
        if self.prev_est_us is not None:
            dt = (m.timestamp - self.prev_est_us) / 1e6
            if 0 < dt < 0.5 and math.isfinite(m.vx) and math.isfinite(m.vy):
                self.moved = self.moved + np.array([m.vy, m.vx]) * dt
        self.prev_est_us = m.timestamp
        if not self.sim_us:
            return
        # filed under the simulation time it arrived at: PX4's own stamps reach ROS converted to
        # wall-clock time, while frames carry simulation time (lockstep keeps the gap to a few ms)
        agl = float(m.dist_bottom) if self.rangefinder and m.dist_bottom_valid else None
        s = State(
            self.sim_us, np.array([m.x, m.y, m.z], float), self.q.copy(), self.moved.copy(), agl
        )
        self.px4_us = m.timestamp
        with self.lock:
            self.history.append(s)
            while self.history and s.t_us - self.history[0].t_us > HISTORY_S * 1e6:
                self.history.popleft()
        self.publish(s)
        self.check_ground(s.height)

    def on_global(self, m: VehicleGlobalPosition) -> None:
        self.global_alt = float(m.alt)

    def on_clock(self, m: Clock) -> None:
        self.sim_us = m.sim.sec * 1_000_000 + m.sim.nsec // 1000

    def check_ground(self, height: float) -> None:
        """Stop the motors once the returning drone sits on the ground.

        In strong wind PX4's own landing detector can miss a touchdown (the drone leans against the
        wind on the ground), and with motors running the wind then drags it. During a camera return,
        once the height has stayed under 0.5 m for 3 s, the node commands a forced disarm. There
        is no stillness test: on the ground PX4's height estimate can keep sinking, which a
        stillness test mistook for flight (development flight, 25 Nov).
        """
        if not self.rtl_sent or self.disarmed:
            return
        on_ground = height < GROUND_HEIGHT_M
        if not on_ground:
            self.ground_since = None
            return
        if self.ground_since is None:
            self.ground_since = self.sim_us
        elif self.sim_us - self.ground_since > GROUND_HOLD_S * 1e6:
            self.command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0, 21196.0)
            self.disarmed = True
            print("on the ground: forced disarm sent", flush=True)

    def on_gps(self, m: SensorGps) -> None:
        # timed on the simulation clock: PX4's stamps as received over the bridge can jump when it
        # re-synchronises, which looked like 10 s of GPS silence (development flight, 25 Nov); the
        # simulation clock only moves forward and stops when the simulator stalls
        if not self.sim_us:
            return
        self.gps_any_us = self.sim_us
        if m.fix_type >= 3:
            self.gps_us = self.sim_us
            self.nofix_since_us = None
        elif self.nofix_since_us is None:
            self.nofix_since_us = self.sim_us

    def gps_ok(self) -> bool:
        since = self.nofix_since_us
        if since is not None and self.sim_us - since > GPS_NOFIX_S * 1e6:
            return False
        return self.sim_us - self.gps_any_us < GPS_SILENT_S * 1e6

    def on_image(self, m: ImgMsg) -> None:
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)[..., :3]
        grey = (a @ np.array([0.299, 0.587, 0.114])).astype(np.uint8)
        t_us = m.header.stamp.sec * 1_000_000 + m.header.stamp.nsec // 1000
        with self.lock:
            self.frame = (t_us, grey)

    def on_poses(self, m: Pose_V) -> None:
        t_us = m.header.stamp.sec * 1_000_000 + m.header.stamp.nsec // 1000
        if self.truth and t_us - self.truth[-1][0] < 50_000:
            return  # keep one true position per 50 ms of simulation time
        for p in m.pose:
            if p.name.startswith("x500"):
                self.truth.append((t_us, p.position.x, p.position.y))

    def state_at(self, t_us: int) -> State | None:
        with self.lock:
            hist = list(self.history)
        if not hist or t_us < hist[0].t_us or t_us - hist[-1].t_us > 200_000:
            return None
        i = bisect.bisect_left([h.t_us for h in hist], t_us)
        return hist[min(i, len(hist) - 1)]

    # logic -----------------------------------------------------------------------------------
    def step(self) -> None:
        with self.lock:
            frame = self.frame
            latest = self.history[-1] if self.history else None
        if latest is None or frame is None or frame[0] <= self.used_frame_us or not self.gps_us:
            return  # (no GPS seen yet: before take-off)
        st = self.state_at(frame[0] - self.lag_us)
        if self.frames_seen < 5:  # clock check, once at the start
            print(f"image {frame[0]} us, PX4 latest {latest.t_us} us, matched {st is not None}",
                  flush=True)  # fmt: skip
        if st is None:
            self.frames_seen += 1 if self.frames_seen < 5 else 0
            return
        self.used_frame_us = frame[0]
        self.frames_seen += 1
        if self.record is not None:
            self.save_frame(frame, st)
        gps_ok = self.gps_ok()
        if gps_ok and not self.lost:
            self.prev = (frame[1], st, st.en)
            self.maybe_keyframe(frame[1], st)
            if self.validate and self.frames_seen % CHECK_EVERY == 0:
                fix = self.locate(frame[1], st, st.en, max_age_us=st.t_us - CHECK_MIN_AGE_US)
                if fix is not None:
                    self.write("check", st, fix[0], fix[1])
            return
        if not self.lost:
            self.lost = True
            self.vo_pos = self.prev[2] if self.prev is not None else st.en
            self.write("gps_lost", st, None, None)
            # the chained camera position starts from the last GPS-based estimate, so the return
            # can begin at once; with --return-height the drone first comes down over the spot,
            # because camera fixes are far better lower down (results/phase5/
            # camera_fix_consistency_study.json: median 5.7 m at 80 m and up, 3.5 m at 40-60 m)
            self.rtl_sent = True
            if (
                self.return_height
                and st.height > self.return_height + 5
                and self.global_alt is not None
            ):
                target = self.global_alt - st.height + self.return_height
                nan = float("nan")
                self.command(VehicleCommand.VEHICLE_CMD_DO_REPOSITION, -1.0, 1.0, nan, nan, target)
                self.descending = True
                self.write("descend_requested", st, None, None)
            else:
                self.command(VehicleCommand.VEHICLE_CMD_NAV_RETURN_TO_LAUNCH)
                self.write("return_requested", st, None, None)
        if self.descending and st.height <= self.return_height + 3:
            self.descending = False
            self.command(VehicleCommand.VEHICLE_CMD_NAV_RETURN_TO_LAUNCH)
            self.write("return_requested", st, None, None)
        self.odometry_step(frame[1], st)
        fix = self.locate(frame[1], st, self.vo_pos)
        if fix is not None and not self.agrees(fix[0]):
            self.write("fix_rejected", st, fix[0], fix[1])
            self.pending = fix[0] - self.vo_pos  # the correction this fix implies
            fix = None
        if fix is None:
            self.write("no_fix", st, None, None)
        else:
            self.pending = None
            self.last_fix = (st.t_us, fix[0])
            self.vo_pos, self.vo_sigma = fix[0], FIX_SIGMA_M
            self.write("fix", st, fix[0], fix[1])
        self.write("vo", st, self.vo_pos, None)
        stale = self.last_fix is None or st.t_us - self.last_fix[0] > NO_FIX_LAND_S * 1e6
        if not self.vision_stopped and st.height < LAND_IN_PLACE_BELOW_M and stale:
            self.vision_stopped = True
            self.write("vision_stopped", st, None, None)
            print("final descent without fixes: positions stopped, PX4 lands in place", flush=True)
        self.anchor = (self.vo_pos, st.moved, st.t_us, self.vo_sigma)
        self.prev = (frame[1], st, self.vo_pos)

    def agrees(self, fix: np.ndarray) -> bool:
        """Whether a new fix fits the chained position, or implies the same correction as the last
        rejected fix."""
        if float(np.linalg.norm(fix - self.vo_pos)) <= max(FIX_AGREE_M, 3 * self.vo_sigma):
            return True
        correction = fix - self.vo_pos
        return (
            self.pending is not None
            and float(np.linalg.norm(correction - self.pending)) <= FIX_PAIR_M
        )

    def odometry_step(self, grey: np.ndarray, st: State) -> None:
        """Move the chained position by the ground shift between the previous frame and this one."""
        if self.prev is None:
            return
        pgrey, pst, ppos = self.prev
        reg = register(pgrey, grey)
        if reg is None:
            # hold the last camera position and widen its uncertainty: falling back on PX4's
            # velocity fed PX4's own drift back into it
            self.vo_sigma = math.hypot(self.vo_sigma, VO_SIGMA_MISS_M)
            return
        centre = ppos + pst.offset_en
        pmap = frame_to_map(centre[0], centre[1], pst.slant, pst.theta, self.fov)
        here = reg[0] @ np.array([WIDTH_PX / 2, HEIGHT_PX / 2, 1.0])
        ground = pmap @ np.array([*here, 1.0])  # ground point at this frame's centre
        step = ground - st.offset_en - ppos
        # a step faster than the drone can fly is a wrong frame-to-frame match: one such step
        # jumped the chain 90 m (development flight, 25 Nov). Not checked against PX4's own
        # movement: once PX4 had drifted, that check rejected every correct step (25 Nov)
        dt = max((st.t_us - pst.t_us) / 1e6, 0.0)
        if float(np.linalg.norm(step)) > VO_MAX_SPEED_M_S * dt + VO_STEP_SLACK_M:
            self.vo_sigma = math.hypot(self.vo_sigma, VO_SIGMA_MISS_M)
            return
        self.vo_pos = ppos + step
        self.vo_sigma = math.hypot(self.vo_sigma, VO_SIGMA_STEP_M)

    def locate(
        self, grey: np.ndarray, st: State, prior: np.ndarray, max_age_us: int | None = None
    ) -> tuple[np.ndarray, int] | None:
        """Drone position (east, north) from one frame, and the inlier count, or None."""
        similar = [
            k
            for k, (_, _, kh, kt) in zip(self.keyframes, self.kf_info, strict=True)
            if 1 / MATCH_HEIGHT_RATIO <= kh / max(st.height, 1.0) <= MATCH_HEIGHT_RATIO
            and (max_age_us is None or kt <= max_age_us)
        ]
        # keyframes are found by the position of their centre: search around the prior's centre
        fix = locate(grey, similar, prior + st.offset_en, max_jump_m=MAX_JUMP_M)
        if fix is None:
            return None
        # quality limits from the offline study (results/phase5/camera_fix_consistency_study.json):
        # weak or inconsistent matches are where the wrong fixes come from
        weak = fix.inliers < FIX_MIN_INLIERS or fix.inliers < FIX_MIN_SHARE * max(fix.matches, 1)
        if st.height > QUALITY_ABOVE_M and weak:
            return None
        return fix.east_north - st.offset_en, fix.inliers

    def save_frame(self, frame: tuple[int, np.ndarray], st: State) -> None:
        """Keep every frame with PX4's state and the true position, for offline matcher studies."""
        Image.fromarray(frame[1]).save(self.record / f"{frame[0]}.png")
        te = tn = None
        if self.truth:
            t, te, tn = min(self.truth, key=lambda x: abs(x[0] - st.t_us))
        row = {
            "image_us": frame[0], "state_us": st.t_us, "ned": st.ned.tolist(), "q": st.q.tolist(),
            "moved": st.moved.tolist(), "gps_ok": self.gps_ok(),
            "true_en": None if te is None else [te, tn],
        }  # fmt: skip
        with open(self.record / "frames.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")

    def maybe_keyframe(self, grey: np.ndarray, st: State) -> None:
        if st.height < MIN_KEYFRAME_HEIGHT_M:
            return
        centre = st.en + st.offset_en
        need = NEW_KEYFRAME_SHARE * 2 * st.slant * math.tan(self.fov / 2)
        for ke, kn, kh, _ in self.kf_info:
            if (
                abs(kh - st.height) <= SAME_HEIGHT_SHARE * st.height
                and math.hypot(ke - centre[0], kn - centre[1]) < need
            ):
                return
        self.keyframes.append(
            Keyframe(grey, frame_to_map(centre[0], centre[1], st.slant, st.theta, self.fov))
        )
        self.kf_info.append((float(centre[0]), float(centre[1]), st.height, st.t_us))
        self.write("keyframe", st, None, None)

    # outputs ---------------------------------------------------------------------------------
    def publish(self, s: State) -> None:
        if self.vision_stopped:
            return
        if self.anchor is None or s.t_us - self.last_pub_us < PUBLISH_PERIOD_US:
            return
        fix, moved_then, then_us, sigma = self.anchor
        age = (s.t_us - then_us) / 1e6
        if age > CARRY_MAX_S:
            return
        self.send_fix(
            fix + (s.moved - moved_then),
            float(s.ned[2]),
            math.hypot(sigma, CARRY_SIGMA_M_S * age),
        )
        self.last_pub_us = s.t_us

    def stamp_us(self) -> int:
        """ROS time in microseconds: PX4's DDS client converts it to PX4 time (timesync)."""
        return self.get_clock().now().nanoseconds // 1000

    def send_fix(self, east_north: np.ndarray, z: float, sigma: float) -> None:
        nan = float("nan")
        m = VehicleOdometry()
        # PX4's own latest stamp as received: the DDS bridge converts it back exactly, however fast
        # the simulation runs (wall-clock stamps were misconverted when it ran slowly)
        m.timestamp = m.timestamp_sample = self.px4_us
        m.pose_frame = VehicleOdometry.POSE_FRAME_NED
        # EKF2 needs all three numbers finite to start external-vision fusion; height is not fused
        # (EKF2_EV_CTRL bit 1 off), so PX4's own height estimate is passed through
        m.position = [float(east_north[1]), float(east_north[0]), z]
        m.q = [nan, nan, nan, nan]
        m.velocity_frame = VehicleOdometry.VELOCITY_FRAME_NED
        m.velocity = [nan, nan, nan]
        m.angular_velocity = [nan, nan, nan]
        m.position_variance = [sigma**2, sigma**2, 100.0]
        m.orientation_variance = [nan, nan, nan]
        m.velocity_variance = [nan, nan, nan]
        m.quality = 100
        self.ev_pub.publish(m)

    def command(
        self,
        cmd: int,
        param1: float = 0.0,
        param2: float = 0.0,
        param5: float = 0.0,
        param6: float = 0.0,
        param7: float = 0.0,
    ) -> None:
        m = VehicleCommand()
        m.timestamp = self.px4_us
        m.command = cmd
        m.param1, m.param2 = param1, param2
        m.param5, m.param6, m.param7 = float(param5), float(param6), float(param7)
        m.target_system = m.target_component = 1
        m.source_system = m.source_component = 1
        m.from_external = True
        self.cmd_pub.publish(m)

    def write(self, event: str, st: State, fix: np.ndarray | None, inliers: int | None) -> None:
        te = tn = None
        if self.truth:
            t, te, tn = min(self.truth, key=lambda x: abs(x[0] - st.t_us))
            if abs(t - st.t_us) > 200_000:
                te = tn = None
        r2 = lambda v: None if v is None else round(float(v), 2)  # noqa: E731
        self.out.writerow(
            [round(st.t_us / 1e6, 2), event, r2(st.en[0]), r2(st.en[1]),
             r2(None if fix is None else fix[0]), r2(None if fix is None else fix[1]),
             r2(te), r2(tn), round(st.height, 1), inliers, len(self.keyframes),
             r2(st.offset_en[0]), r2(st.offset_en[1]), self.att_msgs]
        )  # fmt: skip
        self.log.flush()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--world", default="herdsight_farm")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--lag-s", type=float, default=0.0, help="picture age behind the frame stamp")
    ap.add_argument("--record", type=Path, default=None, help="folder to keep every frame in")
    ap.add_argument("--rangefinder", action="store_true", help="use the rangefinder height")
    ap.add_argument(
        "--return-height", type=float, default=0.0, help="come down to this height before returning"
    )
    args = ap.parse_args(argv)
    plan = yaml.safe_load(args.plan.read_text(encoding="utf-8"))
    rclpy.init()
    node = CameraReturn(
        plan,
        args.log,
        args.world,
        args.validate,
        args.lag_s,
        args.record,
        args.rangefinder,
        args.return_height,
    )
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
