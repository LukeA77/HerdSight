#include "herdsight/count.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>
#include <set>

using herdsight::count::Candidates;
using herdsight::count::features;
using herdsight::count::greedy_join;
using herdsight::count::log_odds;
using herdsight::count::Node;
using herdsight::count::Scale;

namespace {

Candidates make(const std::vector<std::array<int, 4>>& rows, const std::vector<double>& d,
                double dt = 2.0) {
  const auto n = static_cast<Eigen::Index>(rows.size());
  Candidates c;
  c.i.resize(n);
  c.a.resize(n);
  c.j.resize(n);
  c.b.resize(n);
  for (Eigen::Index k = 0; k < n; ++k) {
    const auto& r = rows[static_cast<std::size_t>(k)];
    c.i[k] = r[0];
    c.a[k] = r[1];
    c.j[k] = r[2];
    c.b[k] = r[3];
  }
  c.d_m = Eigen::Map<const Eigen::VectorXd>(d.data(), n);
  c.dt_s = Eigen::VectorXd::Constant(n, dt);
  c.geotag = Eigen::VectorXd::Zero(n);
  c.fixed_gate_m = Eigen::VectorXd::Constant(n, 1.0);
  c.size = Eigen::VectorXd::Zero(n);
  c.low_score = Eigen::VectorXd::Zero(n);
  return c;
}

}  // namespace

TEST(Count, CompetitionMarksMutualClosestAndMargin) {
  // still 0 detections 0, 1; still 1 detections 0, 1: 0-0 close, 0-1 farther
  const auto c = make({{0, 0, 1, 0}, {0, 0, 1, 1}, {0, 1, 1, 1}}, {0.1, 1.0, 0.2});
  const auto x = features(c, 0.0, Scale::Fixed);
  EXPECT_DOUBLE_EQ(x(0, 0), 0.01);
  EXPECT_DOUBLE_EQ(x(0, 2), 1.0);  // 0-0 mutual closest
  EXPECT_DOUBLE_EQ(x(1, 2), 0.0);
  EXPECT_NEAR(x(0, 1), 0.9, 1e-12);   // rival of 0-0 is 0-1 at 1.0
  EXPECT_NEAR(x(1, 1), -0.9, 1e-12);  // 0-1 has a closer rival
  EXPECT_NEAR(x(2, 1), 0.8, 1e-12);  // 1-1's rival is 0-1, through detection 1 of still 1
}

TEST(Count, MovementWidensTheGateWithTime) {
  const auto c = make({{0, 0, 1, 0}}, {1.0}, 300.0);
  const double fixed = features(c, 0.002, Scale::Fixed)(0, 0);
  const double moving = features(c, 0.002, Scale::Movement)(0, 0);
  EXPECT_DOUBLE_EQ(fixed, 1.0);
  EXPECT_NEAR(moving, 1.0 / (1.0 + 4 * 0.002 * 300.0), 1e-12);
}

TEST(Count, LogOddsStandardises) {
  Eigen::MatrixXd x(2, 1);
  x << 1.0, 3.0;
  Eigen::VectorXd mean(1), sd(1), coef(2);
  mean << 2.0;
  sd << 1.0;
  coef << 0.5, 2.0;
  const auto lo = log_odds(x, mean, sd, coef);
  EXPECT_DOUBLE_EQ(lo[0], -1.5);
  EXPECT_DOUBLE_EQ(lo[1], 2.5);
  EXPECT_THROW(log_odds(x, mean, sd, Eigen::VectorXd::Zero(3)), std::invalid_argument);
}

TEST(Count, GreedyJoinNeverPutsOneStillTwice) {
  const auto c = make({{0, 0, 1, 0}, {0, 1, 1, 0}, {1, 0, 2, 0}}, {0, 0, 0});
  Eigen::VectorXd w(3);
  w << 2.0, 1.5, 1.0;
  const auto g = greedy_join({2, 1, 1}, c, w);
  ASSERT_EQ(g.size(), 2u);
  for (const auto& grp : g) {
    std::set<int> s;
    for (const auto& n : grp) EXPECT_TRUE(s.insert(n.first).second);
  }
  EXPECT_EQ(g[0], (std::vector<Node>{{0, 0}, {1, 0}, {2, 0}}));
}

TEST(Count, NegativeWeightsAreNotJoined) {
  const auto c = make({{0, 0, 1, 0}}, {0});
  Eigen::VectorXd w(1);
  w << -0.5;
  EXPECT_EQ(greedy_join({1, 1}, c, w).size(), 2u);
}

TEST(Count, BadInputIsRefused) {
  const auto c = make({{0, 3, 1, 0}}, {0});
  Eigen::VectorXd w(1);
  w << 1.0;
  EXPECT_THROW(greedy_join({1, 1}, c, w), std::out_of_range);
  EXPECT_THROW(greedy_join({1, 1}, c, Eigen::VectorXd::Zero(2)), std::invalid_argument);
}
