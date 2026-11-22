// Survey and herd-mode logic (Phase 5 plan, step 7): order, captures and welfare limits.
#include <gtest/gtest.h>

#include <algorithm>

#include "herdsight_mission/survey_logic.hpp"

using herdsight::Limits;
using herdsight::Plan;
using herdsight::SurveyLogic;
using herdsight::TargetKind;

namespace {

Plan line_plan(int n, float height = 83.0f) {
  Plan p;
  p.camp = "test";
  p.height_agl_m = height;
  p.speed_m_s = 8.0f;
  std::vector<herdsight::PhotoPoint> photos;
  for (int i = 0; i < n; ++i) photos.push_back({i, Eigen::Vector3f(10.0f * i, 0.0f, -height)});
  p.sorties.push_back(photos);
  return p;
}

// Fly straight to each target and report arrival.
void fly_all(SurveyLogic& s, std::vector<TargetKind>& captured, float& min_height) {
  for (int guard = 0; guard < 1000 && !s.done(); ++guard) {
    const auto t = s.current();
    min_height = std::min(min_height, -t->ned.z());
    if (auto c = s.update(t->ned)) captured.push_back(c->kind);
  }
}

}  // namespace

TEST(SurveyLogic, VisitsEveryPhotoPointInOrder) {
  SurveyLogic s(line_plan(5), Limits{});
  EXPECT_EQ(s.photosPlanned(), 5);
  int expect = 0;
  while (!s.done()) {
    auto t = s.current();
    ASSERT_EQ(t->photo_id, expect);
    EXPECT_FALSE(s.update(t->ned + Eigen::Vector3f(5, 0, 0)).has_value());  // not there yet
    auto c = s.update(t->ned);
    ASSERT_TRUE(c.has_value());
    EXPECT_EQ(c->photo_id, expect++);
  }
  EXPECT_EQ(s.photosTaken(), 5);
}

TEST(SurveyLogic, HerdModeStaysWithinTheWelfareLimits) {
  Limits lim;
  lim.frame_along_m = 90.0f;
  SurveyLogic s(line_plan(6), lim);
  s.update(s.current()->ned);  // first photo
  ASSERT_TRUE(s.herdFound(Eigen::Vector2f(25.0f, 40.0f), 30.0f));
  EXPECT_FALSE(s.herdFound(Eigen::Vector2f(25.0f, 40.0f), 30.0f));  // already queued
  // the descent point is outside the herd's edge plus the clearance radius
  const auto approach = s.current()->ned;
  const float off = (Eigen::Vector2f(approach.x(), approach.y()) - Eigen::Vector2f(25, 40)).norm();
  EXPECT_GE(off, 30.0f / 2 + lim.clearance_m - 1e-3f);
  EXPECT_FLOAT_EQ(-approach.z(), 83.0f);  // reached at survey height
  std::vector<TargetKind> captured;
  float min_h = 1e9f;
  fly_all(s, captured, min_h);
  EXPECT_GE(min_h, lim.herd_mode_height_m - 1e-3f);
  EXPECT_EQ(s.herdStills(), 1);
  EXPECT_EQ(s.photosTaken(), 6);
  EXPECT_EQ(std::count(captured.begin(), captured.end(), TargetKind::HerdPhoto), 1);
}

TEST(SurveyLogic, NoHerdModeWhenTheHerdFitsASurveyStill) {
  Limits lim;
  lim.frame_along_m = 90.0f;
  SurveyLogic s(line_plan(3), lim);
  EXPECT_FALSE(s.herdFound(Eigen::Vector2f(0, 0), 80.0f));  // would need to fly higher
  EXPECT_EQ(s.photosPlanned(), 3);
}
