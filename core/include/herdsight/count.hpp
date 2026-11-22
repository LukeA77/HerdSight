// Counting across overlapping stills: pair evidence, the same-animal score and greedy joining.
//
// C++ port of the method Phase 3 kept (docs/plans/phase3-counting-across-images.md). Python
// references: herdsight.count.evidence (features, Logistic.log_odds) and
// herdsight.count.clustering.greedy_join with herdsight.count.identities.constrained_identities.
// tests/test_core_parity.py checks each against its reference on recorded flight data.
#pragma once

#include <Eigen/Core>
#include <utility>
#include <vector>

namespace herdsight::count {

// One row per candidate pair: detection a of still i and detection b of still j.
struct Candidates {
  Eigen::VectorXi i, a, j, b;
  Eigen::VectorXd d_m;           // distance on the ground (metres)
  Eigen::VectorXd dt_s;          // time between the two stills
  Eigen::VectorXd geotag;        // 1 where registration failed (geotag fallback)
  Eigen::VectorXd fixed_gate_m;  // Phase 1 gate for this pair of stills
  Eigen::VectorXd size;          // |log(length_a / length_b)|
  Eigen::VectorXd low_score;     // 1 - min(score_a, score_b)
};

enum class Scale { Fixed, Movement };

inline constexpr double kMarginCap = 5.0;

// Feature matrix, columns dist2, margin, mutual, size, low_score, geotag (FEATURES in Python).
// Movement widens the gate by the grazing allowance: scale^2 = gate^2 + 4 D dt.
Eigen::MatrixXd features(const Candidates& c, double diffusion_m2_s, Scale scale);

// Log odds of a logistic model on standardised features (coef[0] is the intercept).
Eigen::VectorXd log_odds(const Eigen::MatrixXd& x, const Eigen::VectorXd& mean,
                         const Eigen::VectorXd& std, const Eigen::VectorXd& coef);

using Node = std::pair<int, int>;  // (still, detection)

// Greedy joining: pairs of positive weight, strongest first (ties in input order), never two
// detections of one still in a group. Returns the groups, each sorted, sorted by first member.
std::vector<std::vector<Node>> greedy_join(const std::vector<int>& counts, const Candidates& c,
                                           const Eigen::VectorXd& weight);

}  // namespace herdsight::count
