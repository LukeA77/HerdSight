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
    EXPECT_FALSE(s.update(t->ned - Eigen::Vector3f(5, 0, 0)).has_value());  // not there yet
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

namespace {

// Two survey lines (north along east = 0, back along east = 30), photo points 10 m apart.
Plan two_line_plan(float height = 83.0f) {
  Plan p = line_plan(0, height);
  p.sorties.clear();
  std::vector<herdsight::PhotoPoint> photos;
  int id = 0;
  for (int i = 0; i < 4; ++i) photos.push_back({id++, Eigen::Vector3f(10.0f * i, 0.0f, -height)});
  for (int i = 3; i >= 0; --i) photos.push_back({id++, Eigen::Vector3f(10.0f * i, 30.0f, -height)});
  p.sorties.push_back(photos);
  return p;
}

// Move towards flyTo() in small steps, as the drone would, and record every capture.
struct Flight {
  std::vector<int> ids;
  std::vector<Eigen::Vector3f> where;
  std::vector<Eigen::Vector3f> sent;
};
Flight fly_steps(SurveyLogic& s, Eigen::Vector3f pos, float step = 0.3f) {
  Flight f;
  for (int guard = 0; guard < 100000 && !s.done(); ++guard) {
    if (auto c = s.update(pos)) {
      f.ids.push_back(c->photo_id);
      f.where.push_back(pos);
    }
    const auto t = s.flyTo(pos);
    if (!t) break;
    if (f.sent.empty() || (f.sent.back() - t->ned).norm() > 1e-3f) f.sent.push_back(t->ned);
    const Eigen::Vector3f d = t->ned - pos;
    pos += d.norm() <= step ? d : Eigen::Vector3f(d.normalized() * step);
  }
  return f;
}

}  // namespace

TEST(SurveyLogic, InPassingEveryStillIsTakenAtItsPointInOrder) {
  SurveyLogic s(two_line_plan(), Limits{});
  const Flight f = fly_steps(s, Eigen::Vector3f(-5.0f, 0.0f, -83.0f));
  ASSERT_EQ(f.ids.size(), 8u);
  for (int i = 0; i < 8; ++i) EXPECT_EQ(f.ids[i], i);
  const auto plan = two_line_plan().sorties[0];
  for (int i = 0; i < 8; ++i) EXPECT_LE((f.where[i] - plan[i].ned).norm(), 1.5f) << i;
}

TEST(SurveyLogic, InPassingTheDroneIsSentToTheEndOfEachLine) {
  SurveyLogic s(two_line_plan(), Limits{});
  const Flight f = fly_steps(s, Eigen::Vector3f(-5.0f, 0.0f, -83.0f));
  // first point, end of line 1, start of line 2 (the turn), end of line 2: never a middle point
  const auto plan = two_line_plan().sorties[0];
  ASSERT_EQ(f.sent.size(), 4u);
  EXPECT_TRUE(f.sent[0].isApprox(plan[0].ned));
  EXPECT_TRUE(f.sent[1].isApprox(plan[3].ned));
  EXPECT_TRUE(f.sent[2].isApprox(plan[4].ned));
  EXPECT_TRUE(f.sent[3].isApprox(plan[7].ned));
}

TEST(SurveyLogic, StoppingAtEachPointWhenPassingIsOff) {
  Limits lim;
  lim.pass_through = false;
  SurveyLogic s(two_line_plan(), lim);
  const Flight f = fly_steps(s, Eigen::Vector3f(-5.0f, 0.0f, -83.0f));
  EXPECT_EQ(f.ids.size(), 8u);
  EXPECT_EQ(f.sent.size(), 8u);  // every photo point is a target
}

TEST(SurveyLogic, NoStillWhenTheDronePassesFarFromThePoint) {
  SurveyLogic s(line_plan(3), Limits{});
  ASSERT_TRUE(s.update(s.current()->ned).has_value());  // first point; the line now starts
  // level with the second point, but 5 m to the side
  EXPECT_FALSE(s.update(Eigen::Vector3f(12.0f, 5.0f, -83.0f)).has_value());
  EXPECT_TRUE(s.update(Eigen::Vector3f(10.5f, 1.0f, -83.0f)).has_value());
}

TEST(SurveyLogic, InPassingHerdModeStillStopsAtEachTarget) {
  Limits lim;
  lim.frame_along_m = 90.0f;
  SurveyLogic s(line_plan(6), lim);
  s.update(s.current()->ned);
  s.update(s.current()->ned);  // on the line now
  ASSERT_TRUE(s.herdFound(Eigen::Vector2f(25.0f, 40.0f), 30.0f));
  // the next place to fly is the herd-mode approach, not the end of the survey line
  const auto pos = s.current()->ned;
  EXPECT_TRUE(s.flyTo(Eigen::Vector3f(10.0f, 0.0f, -83.0f))->ned.isApprox(pos));
  float min_h = 1e9f;
  const Flight f = fly_steps(s, Eigen::Vector3f(10.0f, 0.0f, -83.0f));
  for (const auto& t : f.sent) min_h = std::min(min_h, -t.z());
  EXPECT_GE(min_h, lim.herd_mode_height_m - 1e-3f);
  EXPECT_EQ(s.herdStills(), 1);
  EXPECT_EQ(s.photosTaken(), 6);
}

TEST(SurveyLogic, DescentEndsBeforeMovingInOverTheHerd) {
  Limits lim;
  lim.frame_along_m = 90.0f;
  SurveyLogic s(line_plan(6), lim);
  s.update(s.current()->ned);
  ASSERT_TRUE(s.herdFound(Eigen::Vector2f(25.0f, 40.0f), 30.0f));
  s.update(s.current()->ned);  // approach reached at survey height
  const auto descent = s.current();
  ASSERT_EQ(descent->kind, TargetKind::Descent);
  const float off = (descent->ned.head<2>() - Eigen::Vector2f(25, 40)).norm();
  EXPECT_NEAR(off, 30.0f / 2 + lim.clearance_m + lim.descent_margin_m, 1e-3f);
  // 1 m above the descent height: within reached_m, but the descent is not finished
  s.update(descent->ned + Eigen::Vector3f(0, 0, -1.0f));
  EXPECT_EQ(s.current()->kind, TargetKind::Descent);
  s.update(descent->ned + Eigen::Vector3f(0.5f, 0, -0.2f));
  EXPECT_EQ(s.current()->kind, TargetKind::HerdPhoto);
}
