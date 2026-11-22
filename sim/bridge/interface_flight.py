"""Fly or watch one survey through a PX4 interface, for the D-41 comparison. Simulation only.

Usage (inside the simulation image, PX4 and the uXRCE-DDS agent running):
  python3 sim/bridge/interface_flight.py --method custom|offboard|mission --plan plan.yaml \
      --out result.json [--crash-after 22] [--limit 900]

One logger measures every method: PX4's position estimate, its navigation state and the stills.
  custom    herdsight_mission (PX4 custom mode) flies; this script only watches its stills;
  offboard  this script streams position setpoints along the survey path;
  mission   this script uploads the survey as a PX4 mission with distance camera triggering,
            then only watches.
With --crash-after N the companion software is lost after N stills: the custom-mode node is
killed, the offboard setpoints stop, and for the mission the watching script's commands stop
(PX4 holds the mission itself). The result records what PX4 did next.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from herdsight_msgs.msg import StillCaptured
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)
from pymavlink import mavutil
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

CAPTURE_M = 3.0  # as SurveyLogic: largest distance off the line for a still
REACHED_M = 1.5
LOOKAHEAD_M = 6.0
CORNER_ACCEL = 2.0  # m/s^2, slowing for a corner
NAV_OFFBOARD = 14
EARTH_M = 6371000.0


def survey_lines(points: np.ndarray) -> list[np.ndarray]:
    """Split the ordered photo points into straight lines (direction change above 5 degrees)."""
    lines, start = [], 0
    for k in range(1, len(points) - 1):
        u, v = points[k] - points[k - 1], points[k + 1] - points[k]
        if u @ v / (np.linalg.norm(u) * np.linalg.norm(v)) < 0.996:
            lines.append(points[start : k + 1])
            start = k + 1
    lines.append(points[start:])
    return lines


class Flight(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("herdsight_interface_flight")
        self.args = args
        plan = yaml.safe_load(Path(args.plan).read_text())
        self.points = np.array([p["ned"] for p in plan["sorties"][0]["photos"]], float)
        self.speed = float(plan["speed_m_s"])
        self.yaw = float(plan["yaw_ned_rad"])
        self.pos: np.ndarray | None = None
        self.nav_state = -1
        self.armed = False
        self.was_armed = False
        self.stills: list[dict] = []
        self.states: list[tuple[float, int]] = []
        self.crash_t: float | None = None
        self.max_after_crash = 0.0
        self.pos_at_crash: np.ndarray | None = None
        self.lowest_after_crash = math.inf
        self.back_over_launch = False
        self.t0 = time.monotonic()
        self.done_t: float | None = None
        self.final_pos: np.ndarray | None = None
        self.lock = threading.Lock()
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=5
        )
        self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1", self.on_pos, qos
        )
        self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status_v1", self.on_status, qos)
        if args.method == "custom":
            self.create_subscription(StillCaptured, "herdsight/still", self.on_custom_still, 10)
        if args.method == "offboard":
            self.mode_pub = self.create_publisher(
                OffboardControlMode, "/fmu/in/offboard_control_mode", qos
            )
            self.sp_pub = self.create_publisher(
                TrajectorySetpoint, "/fmu/in/trajectory_setpoint", qos
            )
            self.cmd_pub = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", qos)
            self.setup_offboard()
            self.create_timer(0.02, self.offboard_tick)

    def now(self) -> float:
        return time.monotonic() - self.t0

    def on_pos(self, m: VehicleLocalPosition) -> None:
        self.pos = np.array([m.x, m.y, m.z], float)
        if self.crash_t is not None and self.pos_at_crash is not None:
            self.max_after_crash = max(
                self.max_after_crash, float(np.linalg.norm(self.pos[:2] - self.pos_at_crash[:2]))
            )
            # height until the drone is back over the launch point (landing is not a height loss)
            if np.linalg.norm(self.pos[:2]) <= 5.0:
                self.back_over_launch = True
            if not self.back_over_launch:
                self.lowest_after_crash = min(self.lowest_after_crash, float(-self.pos[2]))

    def on_status(self, m: VehicleStatus) -> None:
        if m.nav_state != self.nav_state:
            self.nav_state = int(m.nav_state)
            self.states.append((round(self.now(), 2), self.nav_state))
        self.armed = m.arming_state == VehicleStatus.ARMING_STATE_ARMED
        self.was_armed |= self.armed
        if self.was_armed and not self.armed and self.done_t is None:
            self.done_t = self.now()
            self.final_pos = None if self.pos is None else self.pos.copy()

    def record_still(self, idx: int | None = None) -> None:
        if self.pos is None:
            return
        with self.lock:
            n = len(self.stills)
            d = np.linalg.norm(self.points[:, :2] - self.pos[:2], axis=1)
            near = int(np.argmin(d))
            self.stills.append(
                {
                    "t": round(self.now(), 3),
                    "index": near if idx is None else idx,
                    "distance_to_nearest_planned_m": float(d[near]),
                    "height_m": float(-self.pos[2]),
                }
            )
            if self.args.crash_after and n + 1 == self.args.crash_after:
                self.crash()

    def on_custom_still(self, m: StillCaptured) -> None:
        self.record_still(int(m.index))

    def crash(self) -> None:
        if self.crash_t is not None:
            return
        self.crash_t = self.now()
        self.pos_at_crash = None if self.pos is None else self.pos.copy()
        if self.args.method == "custom":
            subprocess.run(["pkill", "-9", "-f", "mission_node"], check=False)

    # --- offboard -------------------------------------------------------------------------
    def setup_offboard(self) -> None:
        h = -float(self.points[0, 2])
        pts = np.vstack([[0.0, 0.0, -h], self.points])
        self.path = pts
        self.seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        self.cum = np.r_[0.0, np.cumsum(self.seg_len)]
        dirs = np.diff(pts, axis=0) / np.maximum(self.seg_len[:, None], 1e-6)
        corners = [self.cum[k + 1] for k in range(len(dirs) - 1) if dirs[k] @ dirs[k + 1] < 0.996]
        self.corners = np.array([*corners, self.cum[-1]])
        self.s = 0.0
        self.ticks = 0
        self.next_photo = 0
        self.last_tick = time.monotonic()
        self.rtl_sent = False

    def command(self, cmd: int, p1: float = 0.0, p2: float = 0.0) -> None:
        m = VehicleCommand()
        m.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        m.command, m.param1, m.param2 = cmd, float(p1), float(p2)
        m.target_system, m.target_component = 1, 1
        m.source_system, m.source_component = 1, 1
        m.from_external = True
        self.cmd_pub.publish(m)

    def carrot(self) -> tuple[np.ndarray, np.ndarray]:
        k = min(int(np.searchsorted(self.cum, self.s, side="right")) - 1, len(self.seg_len) - 1)
        u = (self.path[k + 1] - self.path[k]) / max(self.seg_len[k], 1e-6)
        return self.path[k] + u * (self.s - self.cum[k]), u

    def offboard_tick(self) -> None:
        if self.crash_t is not None or self.rtl_sent:
            return  # the companion is gone, or PX4 is flying the return: nothing more is sent
        now = time.monotonic()
        dt, self.last_tick = now - self.last_tick, now
        self.ticks += 1
        stamp = int(self.get_clock().now().nanoseconds / 1000)
        mode = OffboardControlMode()
        mode.timestamp, mode.position, mode.velocity = stamp, True, True
        self.mode_pub.publish(mode)
        if self.pos is None:
            return
        sp = TrajectorySetpoint()
        sp.timestamp = stamp
        sp.yaw = self.yaw
        climbing = self.pos[2] > self.path[0, 2] + 2.0
        if climbing:
            sp.position = [0.0, 0.0, float(self.path[0, 2])]
            sp.velocity = [math.nan, math.nan, math.nan]
        else:
            carrot, u = self.carrot()
            ahead = self.corners[self.corners > self.s + 1e-3]
            to_corner = float(ahead[0] - self.s) if len(ahead) else 0.0
            v = min(self.speed, math.sqrt(2 * CORNER_ACCEL * to_corner) + 0.5)
            if np.linalg.norm(self.pos - carrot) < LOOKAHEAD_M:
                self.s = min(self.s + v * dt, float(self.cum[-1]))
            carrot, u = self.carrot()
            sp.position = [float(x) for x in carrot]
            sp.velocity = [float(x) for x in u * v]
            self.capture()
        self.sp_pub.publish(sp)
        if self.ticks == 60:
            self.command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1, 6)
        if self.ticks >= 80 and not self.armed and not self.was_armed and self.ticks % 50 == 0:
            self.command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1, 6)
            self.command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1)
        if self.next_photo >= len(self.points) and not self.rtl_sent:
            self.command(VehicleCommand.VEHICLE_CMD_NAV_RETURN_TO_LAUNCH)
            self.rtl_sent = True

    def capture(self) -> None:
        if self.next_photo >= len(self.points):
            return
        t = self.points[self.next_photo]
        off = self.pos - t
        hit = np.linalg.norm(off) <= REACHED_M
        if not hit and self.next_photo:
            leg = t - self.points[self.next_photo - 1]
            u = leg / max(np.linalg.norm(leg), 1e-6)
            along = off @ u
            hit = along >= 0 and np.linalg.norm(off - along * u) <= CAPTURE_M
        if hit:
            self.record_still(self.next_photo)
            self.next_photo += 1


def mission_thread(node: Flight) -> None:
    """Upload the survey as a mission with distance triggering, start it and log the triggers."""
    link = mavutil.mavlink_connection("udpin:0.0.0.0:14540", source_system=245)
    link.wait_heartbeat(timeout=120)
    gp = link.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=60)
    lat0, lon0 = gp.lat / 1e7, gp.lon / 1e7
    pts = node.points
    h = float(-pts[0, 2])

    def wp(ne: np.ndarray) -> tuple[int, int]:
        lat = lat0 + math.degrees(ne[0] / EARTH_M)
        lon = lon0 + math.degrees(ne[1] / (EARTH_M * math.cos(math.radians(lat0))))
        return int(lat * 1e7), int(lon * 1e7)

    m = mavutil.mavlink
    rel = m.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT
    items: list[tuple] = [(m.MAV_CMD_NAV_TAKEOFF, rel, 0, 0, 0, math.nan, *wp(np.zeros(2)), h)]
    items.append((m.MAV_CMD_DO_CHANGE_SPEED, m.MAV_FRAME_MISSION, 1, node.speed, -1, 0, 0, 0, 0))
    for line in survey_lines(pts):
        step = float(np.median(np.linalg.norm(np.diff(line, axis=0), axis=1)))
        items.append((m.MAV_CMD_NAV_WAYPOINT, rel, 0, REACHED_M, 0, math.nan, *wp(line[0][:2]), h))
        items.append((m.MAV_CMD_DO_SET_CAM_TRIGG_DIST, m.MAV_FRAME_MISSION, step, 0, 1, 0, 0, 0, 0))
        items.append((m.MAV_CMD_NAV_WAYPOINT, rel, 0, REACHED_M, 0, math.nan, *wp(line[-1][:2]), h))
        items.append((m.MAV_CMD_DO_SET_CAM_TRIGG_DIST, m.MAV_FRAME_MISSION, 0, 0, 0, 0, 0, 0, 0))
    items.append((m.MAV_CMD_NAV_RETURN_TO_LAUNCH, m.MAV_FRAME_MISSION, 0, 0, 0, 0, 0, 0, 0))
    link.mav.mission_count_send(1, 1, len(items), m.MAV_MISSION_TYPE_MISSION)
    while True:
        r = link.recv_match(
            type=["MISSION_REQUEST_INT", "MISSION_REQUEST", "MISSION_ACK"],
            blocking=True,
            timeout=20,
        )
        if r is None:
            raise RuntimeError("mission upload timed out")
        if r.get_type() == "MISSION_ACK":
            if r.type != m.MAV_MISSION_ACCEPTED:
                raise RuntimeError(f"mission refused: {r.type}")
            break
        cmd, frame, p1, p2, p3, p4, x, y, z = items[r.seq]
        link.mav.mission_item_int_send(
            1,
            1,
            r.seq,
            frame,
            cmd,
            int(r.seq == 0),
            1,
            p1,
            p2,
            p3,
            p4,
            x,
            y,
            z,
            m.MAV_MISSION_TYPE_MISSION,
        )
    while not node.was_armed and node.now() < node.args.limit:
        link.mav.command_long_send(1, 1, m.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
        time.sleep(0.5)
        link.mav.command_long_send(1, 1, m.MAV_CMD_MISSION_START, 0, 0, 0, 0, 0, 0, 0, 0)
        time.sleep(2.0)
    while node.done_t is None and node.now() < node.args.limit:
        if link.recv_match(type="CAMERA_TRIGGER", blocking=True, timeout=1.0) is not None:
            node.record_still()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=("custom", "offboard", "mission"), required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crash-after", type=int, default=0)
    ap.add_argument("--limit", type=float, default=900.0)
    args = ap.parse_args()
    rclpy.init()
    node = Flight(args)
    if args.method == "mission":
        threading.Thread(target=mission_thread, args=(node,), daemon=True).start()
    end_after_crash = 240.0  # seconds to watch what PX4 does once the companion is lost
    while rclpy.ok() and node.now() < args.limit and node.done_t is None:
        rclpy.spin_once(node, timeout_sec=0.02)
        if node.crash_t is not None and node.now() - node.crash_t > end_after_crash:
            break
    st = sorted(node.stills, key=lambda s: s["t"])
    after = [s for s in node.states if node.crash_t is not None and s[0] >= node.crash_t]
    out = {
        "method": args.method,
        "crash_after_stills": args.crash_after,
        "planned_stills": len(node.points),
        "stills": len(st),
        "distance_to_planned_m_median": float(
            np.median([s["distance_to_nearest_planned_m"] for s in st])
        )
        if st
        else None,
        "distance_to_planned_m_max": max(
            (s["distance_to_nearest_planned_m"] for s in st), default=None
        ),
        "height_m_min": min((s["height_m"] for s in st), default=None),
        "height_m_max": max((s["height_m"] for s in st), default=None),
        "first_to_last_still_s": st[-1]["t"] - st[0]["t"] if len(st) > 1 else None,
        "landed_and_disarmed": node.done_t is not None,
        "final_distance_from_launch_m": None
        if node.final_pos is None
        else float(np.linalg.norm(node.final_pos[:2])),
        "nav_states": node.states,
        "crash_t": node.crash_t,
        "nav_states_after_crash": after,
        "first_state_change_after_crash_s": None
        if not after or node.crash_t is None
        else after[0][0] - node.crash_t,
        "largest_move_after_crash_m": node.max_after_crash if node.crash_t is not None else None,
        "lowest_height_after_crash_m": None
        if node.crash_t is None or not math.isfinite(node.lowest_after_crash)
        else node.lowest_after_crash,
        "stills_detail": st,
    }
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("stills_detail", "nav_states")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
