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
Repeat: once GPS is lost (no 3D fix for 3 s of PX4 time), each new frame is matched to keyframes
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

# GPS arrives up to 1 s apart in the simulator; one dropped message must not count as loss
GPS_TIMEOUT_S = 3.0
NEW_KEYFRAME_SHARE = 0.4  # of the footprint width
SAME_HEIGHT_SHARE = 0.15
MIN_KEYFRAME_HEIGHT_M = 4.0
MATCH_HEIGHT_RATIO = 2.0  # only keyframes from half to twice the current height
MAX_JUMP_M = 75.0
FIX_SIGMA_M = 1.0
CARRY_SIGMA_M_S = 0.3  # added uncertainty per second of carrying a fix forward
CARRY_MAX_S = 1.0  # only bridges a missed frame; longer carrying fed drift back into PX4
GROUND_HEIGHT_M = 0.5  # below this, with no vertical movement, the drone is on the ground
GROUND_STILL_M_S = 0.2
GROUND_HOLD_S = 3.0
VO_SIGMA_STEP_M = 0.15  # uncertainty added per chained frame-to-frame step
VO_SIGMA_MISS_M = 0.5  # added when a step falls back on PX4's velocity
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

    def __init__(self, t_us: int, ned: np.ndarray, q: np.ndarray, moved: np.ndarray) -> None:
        self.t_us, self.ned, self.q, self.moved = t_us, ned, q, moved
        r = rotation(q)
        axis = r @ np.array([0.0, 0.0, 1.0])  # optical axis (body down) in NED
        self.height = float(-ned[2])
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
    ) -> None:
        super().__init__("camera_return")
        self.fov = 2 * math.atan(float(plan["footprint_m"][0]) / 2 / float(plan["height_agl_m"]))
        self.validate = validate
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
        s = State(self.sim_us, np.array([m.x, m.y, m.z], float), self.q.copy(), self.moved.copy())
        self.px4_us = m.timestamp
        with self.lock:
            self.history.append(s)
            while self.history and s.t_us - self.history[0].t_us > HISTORY_S * 1e6:
                self.history.popleft()
        self.publish(s)
        self.check_ground(m)

    def on_clock(self, m: Clock) -> None:
        self.sim_us = m.sim.sec * 1_000_000 + m.sim.nsec // 1000

    def check_ground(self, m: VehicleLocalPosition) -> None:
        """Stop the motors once the returning drone sits on the ground.

        In strong wind PX4's own landing detector can miss a touchdown (the drone leans against the
        wind on the ground), and with motors running the wind then drags it. During a camera return,
        once the height is under 0.5 m and the drone has moved up or down no faster than 0.2 m/s
        for 3 s, the node commands a forced disarm.
        """
        if not self.rtl_sent or self.disarmed:
            return
        on_ground = -m.z < GROUND_HEIGHT_M and abs(m.vz) < GROUND_STILL_M_S
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
        if m.fix_type >= 3:  # PX4 time, so that a slow simulator does not look like GPS loss
            self.gps_us = m.timestamp

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
        gps_ok = self.px4_us - self.gps_us < GPS_TIMEOUT_S * 1e6
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
            # can begin at once
            self.command(VehicleCommand.VEHICLE_CMD_NAV_RETURN_TO_LAUNCH)
            self.rtl_sent = True
            self.write("return_requested", st, None, None)
        self.odometry_step(frame[1], st)
        fix = self.locate(frame[1], st, self.vo_pos)
        if fix is None:
            self.write("no_fix", st, None, None)
        else:
            self.last_fix = (st.t_us, fix[0])
            self.vo_pos, self.vo_sigma = fix[0], FIX_SIGMA_M
            self.write("fix", st, fix[0], fix[1])
        self.write("vo", st, self.vo_pos, None)
        self.anchor = (self.vo_pos, st.moved, st.t_us, self.vo_sigma)
        self.prev = (frame[1], st, self.vo_pos)

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
        self.vo_pos = ground - st.offset_en
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
        return fix.east_north - st.offset_en, fix.inliers

    def save_frame(self, frame: tuple[int, np.ndarray], st: State) -> None:
        """Keep every frame with PX4's state and the true position, for offline matcher studies."""
        Image.fromarray(frame[1]).save(self.record / f"{frame[0]}.png")
        te = tn = None
        if self.truth:
            t, te, tn = min(self.truth, key=lambda x: abs(x[0] - st.t_us))
        row = {
            "image_us": frame[0], "state_us": st.t_us, "ned": st.ned.tolist(), "q": st.q.tolist(),
            "moved": st.moved.tolist(), "gps_ok": self.px4_us - self.gps_us < GPS_TIMEOUT_S * 1e6,
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

    def command(self, cmd: int, param1: float = 0.0, param2: float = 0.0) -> None:
        m = VehicleCommand()
        m.timestamp = self.px4_us
        m.command = cmd
        m.param1, m.param2 = param1, param2
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
    args = ap.parse_args(argv)
    plan = yaml.safe_load(args.plan.read_text(encoding="utf-8"))
    rclpy.init()
    node = CameraReturn(plan, args.log, args.world, args.validate, args.lag_s, args.record)
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
