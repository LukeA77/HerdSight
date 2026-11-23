#!/bin/bash
# One simulated survey flight, headless (Phase 5 plan, step 8). Simulation only.
#
# Runs inside the simulation image (sim/docker/Dockerfile), with this repository at /repo and the
# built ROS 2 workspace (ros2/) at /ws:
#   docker run --rm -v <repo>:/repo:ro -v hs_ros_ws:/ws -v <out>:/out herdsight-sim \
#     bash /repo/sim/bridge/run_flight.sh [seconds] [seed]
#
# Steps: write the farm world, herd and plan (make_world.py); start the uXRCE-DDS agent; start
# PX4 SITL with Gazebo (no window) in that world; set the parameters a bench simulation needs (no
# radio control or ground station is connected); start the herd bridge and the mission node with
# autostart on; stop everything after the time limit. Logs and the plan are left in /out.
# ROS setup files read unset variables, so they are sourced before `set -u`
. /opt/ros/jazzy/setup.bash
. /opt/px4_ws/install/setup.bash
. /ws/install/setup.bash
set -u
SECONDS_LIMIT=${1:-900}
SEED=${2:-0}
OUT=${OUT:-/out}
# offboard_mission_node (D-41) or mission_node (custom mode)
MISSION_NODE=${MISSION_NODE:-offboard_mission_node}
# failure and herd-mode checks, after 15 stills: INJECT=herd (a herd is reported), INJECT=crash
# (the mission node is killed), INJECT=gnss (the GPS fix is removed) or INJECT=link (the ground
# station stops); none by default. Geofence: run with PX4_PARAM_GF_ACTION and GF_MAX_HOR_DIST.
INJECT=${INJECT:-none}
mkdir -p "$OUT"

python3 /repo/sim/bridge/make_world.py --out "$OUT" --seed "$SEED" | tee "$OUT/world.txt"
cp "$OUT/herdsight_farm.sdf" "$PX4_DIR/Tools/simulation/gz/worlds/herdsight_farm.sdf"

MicroXRCEAgent udp4 -p 8888 > "$OUT/agent.log" 2>&1 &
pids=($!)

cd "$PX4_DIR"
HEADLESS=1 PX4_GZ_WORLD=herdsight_farm PX4_SIM_MODEL=gz_x500 PX4_SYS_AUTOSTART=4001 \
  ./build/px4_sitl_default/bin/px4 -d > "$OUT/px4.log" 2>&1 &
pids+=($!)

# wait until PX4's start-up script has finished (or give up after 3 minutes)
for _ in $(seq 180); do
  grep -q "Startup script returned successfully" "$OUT/px4.log" && break
  sleep 1
done
if ! grep -q "Startup script returned successfully" "$OUT/px4.log"; then
  echo "PX4 did not start; see $OUT/px4.log"
  kill "${pids[@]}" 2>/dev/null
  exit 1
fi

# bench simulation: no radio control and no ground station are connected. The mission itself
# waits until PX4 reports it is ready to arm (behaviour tree, WaitReadyToArm).
P=./build/px4_sitl_default/bin
$P/px4-param set COM_RC_IN_MODE 4
$P/px4-param set NAV_DLL_ACT 0
$P/px4-param set NAV_RCL_ACT 0
# if the offboard setpoints stop, PX4 returns to launch (D-41)
$P/px4-param set COM_OBL_RC_ACT 3
# act on a failsafe at once: the default 5 s delay is for a pilot, and no pilot is assumed here
$P/px4-param set COM_FAIL_ACT_T 0
# PX4 caps the mission's requested speed at its cruise speed (MPC_XY_CRUISE, default 5 m/s), so
# the cruise speed is set to the plan's survey speed
SPEED=$(python3 -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['speed_m_s'])" "$OUT/plan.yaml")
$P/px4-param set MPC_XY_CRUISE "$SPEED"

if [ "$INJECT" = link ]; then  # a ground station that sends heartbeats until the link is cut
  $P/px4-param set NAV_DLL_ACT 2
  $P/px4-param set COM_DL_LOSS_T 10
  python3 -c "
import time
from pymavlink import mavutil
m = mavutil.mavlink_connection('udpout:127.0.0.1:18570', source_system=255)
while True:
    m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
    time.sleep(0.5)
" &
  GCS_PID=$!
  pids+=($GCS_PID)
  sleep 3
fi
python3 /repo/sim/bridge/herd_bridge.py --state "$OUT/herd_state.pkl" --seed "$SEED" \
  > "$OUT/herd_bridge.log" 2>&1 &
pids+=($!)
FRAME_ALONG=$(python3 -c "import sys, yaml; print(float(yaml.safe_load(open(sys.argv[1]))['footprint_m'][1]))" "$OUT/plan.yaml")
ros2 run herdsight_mission "$MISSION_NODE" --ros-args -p plan_file:="$OUT/plan.yaml" \
  -p autostart:=true -p frame_along_m:="$FRAME_ALONG" > "$OUT/mission.log" 2>&1 &
pids+=($!)
ros2 topic echo --csv herdsight/still herdsight_msgs/msg/StillCaptured > "$OUT/stills.csv" 2>&1 &
pids+=($!)

# Gazebo's real-time factor every 30 s (a slow computer runs the simulation slower than real time)
end=$((SECONDS + SECONDS_LIMIT))
next_stats=0
while [ $SECONDS -lt $end ]; do
  grep -q "mission finished: SUCCESS" "$OUT/mission.log" && break
  grep -q "Disarmed by landing" "$OUT/px4.log" && break
  if [ "$INJECT" != none ] && [ "$(grep -c . "$OUT/stills.csv")" -ge 15 ]; then
    if [ "$INJECT" = herd ]; then
      ros2 topic pub --once herdsight/herd_found herdsight_msgs/msg/HerdFound         "{centre: {x: 120.0, y: 0.0, z: 0.0}, extent_m: 20.0, animals_estimate: 20}" > /dev/null
    elif [ "$INJECT" = gnss ]; then
      $P/px4-param set SIM_GPS_USED 0
    elif [ "$INJECT" = link ]; then
      kill "$GCS_PID"
    else
      pkill -9 -f "$MISSION_NODE"
    fi
    echo "injected $INJECT at ${SECONDS} s"
    INJECT=none
  fi
  if [ $SECONDS -ge $next_stats ]; then
    rtf=$(timeout 5 gz topic -e -t /world/herdsight_farm/stats -n 1 2>/dev/null       | awk '/real_time_factor/ {print $2}')
    echo "$SECONDS,${rtf:-}" >> "$OUT/real_time_factor.csv"
    next_stats=$((SECONDS + 30))
  fi
  sleep 2
done
sleep 5
kill "${pids[@]}" 2>/dev/null
pkill -f "gz sim" 2>/dev/null
# PX4's own flight log (ULog) for the run
cp -r "$PX4_DIR/build/px4_sitl_default/rootfs/log" "$OUT/px4_ulog" 2>/dev/null
echo "stills published: $(grep -c . "$OUT/stills.csv" || true)"
echo "real-time factor (mean): $(awk -F, '$2 != "" {s += $2; n++} END {if (n) printf "%.2f", s / n}' "$OUT/real_time_factor.csv")"
tail -5 "$OUT/mission.log"
