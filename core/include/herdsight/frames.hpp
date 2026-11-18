// Reference frames and rotations.
//
// Conventions match python/herdsight/geo/frames.py exactly (checked by the parity test):
//   World W: North, East, Down (NED). Body B: forward, right, down (FRD).
//   R_ab rotates a vector from frame b into frame a: v_a = R_ab * v_b.
//   Attitude: aerospace yaw-pitch-roll, R_wb = Rz(yaw) Ry(pitch) Rx(roll). Positive pitch is
//   nose up, positive roll is right wing down, yaw is heading clockwise from North.
#pragma once

#include <Eigen/Core>

namespace herdsight::frames {

struct YawPitchRoll {
  double yaw;    // radians
  double pitch;  // radians, in [-pi/2, pi/2]
  double roll;   // radians
};

// Active right-handed rotations about x, y and z by angle a (radians).
Eigen::Matrix3d rot_x(double a);
Eigen::Matrix3d rot_y(double a);
Eigen::Matrix3d rot_z(double a);

// Body-to-world rotation from aerospace yaw, pitch, roll (radians).
Eigen::Matrix3d R_wb_from_ypr(double yaw, double pitch, double roll);

// Inverse of R_wb_from_ypr. Singular at pitch = +/- pi/2.
YawPitchRoll ypr_from_R_wb(const Eigen::Matrix3d& R_wb);

}  // namespace herdsight::frames
