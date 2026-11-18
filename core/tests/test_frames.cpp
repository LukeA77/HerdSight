#include "herdsight/frames.hpp"

#include <gtest/gtest.h>

#include <Eigen/LU>

#include <numbers>
#include <random>

using herdsight::frames::R_wb_from_ypr;
using herdsight::frames::ypr_from_R_wb;

namespace {
constexpr double kPi = std::numbers::pi;
constexpr double kTol = 1e-12;
}  // namespace

TEST(Frames, ZeroAttitudeIsIdentity) {
  EXPECT_TRUE(R_wb_from_ypr(0.0, 0.0, 0.0).isApprox(Eigen::Matrix3d::Identity(), kTol));
}

TEST(Frames, YawNinetyPointsForwardEast) {
  const Eigen::Vector3d fwd_w = R_wb_from_ypr(kPi / 2, 0.0, 0.0) * Eigen::Vector3d::UnitX();
  EXPECT_NEAR(fwd_w.x(), 0.0, kTol);
  EXPECT_NEAR(fwd_w.y(), 1.0, kTol);
  EXPECT_NEAR(fwd_w.z(), 0.0, kTol);
}

TEST(Frames, PositivePitchIsNoseUp) {
  // In North-East-Down, up is negative Down.
  const Eigen::Vector3d fwd_w = R_wb_from_ypr(0.0, 0.3, 0.0) * Eigen::Vector3d::UnitX();
  EXPECT_LT(fwd_w.z(), 0.0);
}

TEST(Frames, PositiveRollIsRightWingDown) {
  const Eigen::Vector3d right_w = R_wb_from_ypr(0.0, 0.0, 0.3) * Eigen::Vector3d::UnitY();
  EXPECT_GT(right_w.z(), 0.0);
}

TEST(Frames, RotationIsOrthonormalAndRoundTrips) {
  std::mt19937_64 rng(20261118);
  std::uniform_real_distribution<double> yaw(-kPi, kPi);
  std::uniform_real_distribution<double> pitch(-1.5, 1.5);
  std::uniform_real_distribution<double> roll(-kPi, kPi);
  for (int i = 0; i < 1000; ++i) {
    const double y = yaw(rng);
    const double p = pitch(rng);
    const double r = roll(rng);
    const Eigen::Matrix3d R = R_wb_from_ypr(y, p, r);
    EXPECT_TRUE((R.transpose() * R).isApprox(Eigen::Matrix3d::Identity(), kTol));
    EXPECT_NEAR(R.determinant(), 1.0, kTol);
    const auto ypr = ypr_from_R_wb(R);
    EXPECT_NEAR(ypr.yaw, y, 1e-9);
    EXPECT_NEAR(ypr.pitch, p, 1e-9);
    EXPECT_NEAR(ypr.roll, r, 1e-9);
  }
}
