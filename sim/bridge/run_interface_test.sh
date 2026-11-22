#!/bin/bash
# One D-41 comparison flight, headless. Simulation only.
#   docker run --rm -v <repo>:/repo:ro -v hs_ros_ws:/ws -v <out>:/out herdsight-sim \
#     bash /repo/sim/bridge/run_interface_test.sh custom|offboard|mission [crash_after] [seed]
# Same world, plan and PX4 settings as run_flight.sh; sim/bridge/interface_flight.py flies or
# watches and writes /out/result.json.
. /opt/ros/jazzy/setup.bash
. /opt/px4_ws/install/setup.bash
. /ws/install/setup.bash
set -u
METHOD=$1
CRASH=${2:-0}
SEED=${3:-0}
OUT=${OUT:-/out}
mkdir -p "$OUT"

python3 /repo/sim/bridge/make_world.py --out "$OUT" --seed "$SEED" > "$OUT/world.txt"
cp "$OUT/herdsight_farm.sdf" "$PX4_DIR/Tools/simulation/gz/worlds/herdsight_farm.sdf"
MicroXRCEAgent udp4 -p 8888 > "$OUT/agent.log" 2>&1 &
pids=($!)

cd "$PX4_DIR"
if [ "$METHOD" = mission ]; then
  export PX4_PARAM_TRIG_MODE=4 PX4_PARAM_TRIG_INTERFACE=3
fi
HEADLESS=1 PX4_GZ_WORLD=herdsight_farm PX4_SIM_MODEL=gz_x500 PX4_SYS_AUTOSTART=4001 \
  ./build/px4_sitl_default/bin/px4 -d > "$OUT/px4.log" 2>&1 &
pids+=($!)
for _ in $(seq 180); do
  grep -q "Startup script returned successfully" "$OUT/px4.log" && break
  sleep 1
done
grep -q "Startup script returned successfully" "$OUT/px4.log" || { echo "PX4 did not start"; kill "${pids[@]}"; exit 1; }

P=./build/px4_sitl_default/bin
$P/px4-param set COM_RC_IN_MODE 4
$P/px4-param set NAV_DLL_ACT 0
$P/px4-param set NAV_RCL_ACT 0
# if offboard setpoints stop: return to launch (not the default, which assumes a pilot)
$P/px4-param set COM_OBL_RC_ACT 3
SPEED=$(python3 -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['speed_m_s'])" "$OUT/plan.yaml")
$P/px4-param set MPC_XY_CRUISE "$SPEED"
sleep 15  # estimator settles and PX4 reports ready

python3 /repo/sim/bridge/herd_bridge.py --state "$OUT/herd_state.pkl" --seed "$SEED" \
  > "$OUT/herd_bridge.log" 2>&1 &
pids+=($!)
if [ "$METHOD" = custom ]; then
  ros2 run herdsight_mission mission_node --ros-args -p plan_file:="$OUT/plan.yaml" \
    -p autostart:=true > "$OUT/mission.log" 2>&1 &
  pids+=($!)
fi
python3 /repo/sim/bridge/interface_flight.py --method "$METHOD" --plan "$OUT/plan.yaml" \
  --out "$OUT/result.json" --crash-after "$CRASH" 2>&1 | tee "$OUT/flight.log" | tail -1
kill "${pids[@]}" 2>/dev/null
pkill -f "gz sim" 2>/dev/null
exit 0
