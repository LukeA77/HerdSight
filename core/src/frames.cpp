#include "herdsight/frames.hpp"

#include <algorithm>
#include <cmath>

namespace herdsight::frames {

Eigen::Matrix3d rot_x(double a) {
  const double c = std::cos(a);
  const double s = std::sin(a);
  Eigen::Matrix3d R;
  R << 1.0, 0.0, 0.0,
       0.0, c, -s,
       0.0, s, c;
  return R;
}

Eigen::Matrix3d rot_y(double a) {
  const double c = std::cos(a);
  const double s = std::sin(a);
  Eigen::Matrix3d R;
  R << c, 0.0, s,
       0.0, 1.0, 0.0,
       -s, 0.0, c;
  return R;
}

Eigen::Matrix3d rot_z(double a) {
  const double c = std::cos(a);
  const double s = std::sin(a);
  Eigen::Matrix3d R;
  R << c, -s, 0.0,
       s, c, 0.0,
       0.0, 0.0, 1.0;
  return R;
}

Eigen::Matrix3d R_wb_from_ypr(double yaw, double pitch, double roll) {
  return rot_z(yaw) * rot_y(pitch) * rot_x(roll);
}

YawPitchRoll ypr_from_R_wb(const Eigen::Matrix3d& R_wb) {
  return {
      std::atan2(R_wb(1, 0), R_wb(0, 0)),
      std::asin(std::clamp(-R_wb(2, 0), -1.0, 1.0)),
      std::atan2(R_wb(2, 1), R_wb(2, 2)),
  };
}

}  // namespace herdsight::frames
