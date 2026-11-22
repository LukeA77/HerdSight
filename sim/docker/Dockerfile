# HerdSight simulation image (Phase 5 plan, step 8): ROS 2 Jazzy, Gazebo Harmonic, PX4 SITL,
# the uXRCE-DDS agent, px4_msgs, px4-ros2-interface-lib and BehaviorTree.CPP, all pinned.
# Build:  docker build -t herdsight-sim -f sim/docker/Dockerfile sim/docker
# Every external source is pinned to a release tag and its commit; change them together.
FROM ros:jazzy-ros-base-noble

ARG PX4_TAG=v1.17.0
ARG PX4_COMMIT=d6f12ad1c4f70ad3230afd7d86e971421e02fef4
ARG PX4_MSGS_COMMIT=86d8239e962f6939e05c3737784f60c02fa884db
ARG PX4_ROS2_LIB_COMMIT=b6d52ad120f68fbe82c9039f85e52c2a8d868083
ARG XRCE_AGENT_TAG=v2.4.3

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8

# Gazebo Harmonic (the ROS 2 Jazzy pairing), the ROS-Gazebo bridge, BehaviorTree.CPP, build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-ros-gz ros-jazzy-behaviortree-cpp ros-jazzy-launch-testing-ament-cmake \
        ros-jazzy-rosbag2-storage-mcap \
        git cmake ninja-build build-essential python3-pip python3-venv ccache \
        wget curl sudo lsb-release gnupg ca-certificates \
        mesa-utils libegl1 libgl1-mesa-dri xvfb \
    && rm -rf /var/lib/apt/lists/*

# PX4 SITL at the pinned release, built for Gazebo (no flight-controller toolchains)
WORKDIR /opt
RUN git clone --depth 1 --branch ${PX4_TAG} --recurse-submodules --shallow-submodules \
        https://github.com/PX4/PX4-Autopilot.git \
    && test "$(git -C PX4-Autopilot rev-parse HEAD)" = "${PX4_COMMIT}"
# PX4's Python requirements are installed beside Ubuntu's own packages (numpy from apt cannot
# be uninstalled by pip on Ubuntu 24.04), kept to the numpy and setuptools versions that
# ROS 2 Jazzy's own Python packages need
RUN printf 'numpy<1.28\nsetuptools<80\n' > /opt/pip-constraints.txt \
    && PIP_BREAK_SYSTEM_PACKAGES=1 PIP_IGNORE_INSTALLED=1 \
       PIP_CONSTRAINT=/opt/pip-constraints.txt \
       bash PX4-Autopilot/Tools/setup/ubuntu.sh --no-nuttx \
    && rm -rf /var/lib/apt/lists/*
# build only: the gz_x500 target would also start the simulator
RUN cd PX4-Autopilot && make px4_sitl \
    && rm -rf build/px4_sitl_default/external

# uXRCE-DDS agent (the PX4-ROS 2 link), at the version PX4's documentation names
RUN git clone --depth 1 --branch ${XRCE_AGENT_TAG} \
        https://github.com/eProsima/Micro-XRCE-DDS-Agent.git \
    && cmake -S Micro-XRCE-DDS-Agent -B Micro-XRCE-DDS-Agent/build -G Ninja \
    && cmake --build Micro-XRCE-DDS-Agent/build && cmake --install Micro-XRCE-DDS-Agent/build \
    && ldconfig && rm -rf Micro-XRCE-DDS-Agent

# ROS 2 workspace with PX4's messages and the PX4 ROS 2 interface library (custom modes, D-41)
WORKDIR /opt/px4_ws/src
RUN git clone https://github.com/PX4/px4_msgs.git && git -C px4_msgs checkout ${PX4_MSGS_COMMIT} \
    && git clone https://github.com/Auterion/px4-ros2-interface-lib.git \
    && git -C px4-ros2-interface-lib checkout ${PX4_ROS2_LIB_COMMIT}
WORKDIR /opt/px4_ws
RUN . /opt/ros/jazzy/setup.sh && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release \
    && rm -rf build log

ENV PX4_DIR=/opt/PX4-Autopilot
RUN echo ". /opt/ros/jazzy/setup.bash && . /opt/px4_ws/install/setup.bash" >> /root/.bashrc
WORKDIR /work
