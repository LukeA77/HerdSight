#include "herdsight/count.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <tuple>

namespace herdsight::count {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

struct BestTwo {
  double best = kInf;
  double second = kInf;
  void add(double v) {
    if (v < best) {
      second = best;
      best = v;
    } else if (v < second) {
      second = v;
    }
  }
  // The closest rival of a pair with value v among this detection's candidates.
  double rival(double v) const { return v <= best ? second : best; }
};

void check_sizes(const Candidates& c) {
  const auto n = c.i.size();
  if (c.a.size() != n || c.j.size() != n || c.b.size() != n || c.d_m.size() != n ||
      c.dt_s.size() != n || c.geotag.size() != n || c.fixed_gate_m.size() != n ||
      c.size.size() != n || c.low_score.size() != n) {
    throw std::invalid_argument("candidate columns differ in length");
  }
}

}  // namespace

Eigen::MatrixXd features(const Candidates& c, double diffusion_m2_s, Scale scale) {
  check_sizes(c);
  const Eigen::Index n = c.i.size();
  Eigen::VectorXd z(n);
  for (Eigen::Index k = 0; k < n; ++k) {
    double s = c.fixed_gate_m[k];
    if (scale == Scale::Movement) {
      s = std::sqrt(s * s + 4.0 * diffusion_m2_s * std::abs(c.dt_s[k]));
    }
    z[k] = c.d_m[k] / std::max(s, 1e-6);
  }
  // competition: per detection and partner still, the best and second-best candidate
  using Key = std::tuple<int, int, int>;
  std::map<Key, BestTwo> by_a, by_b;
  for (Eigen::Index k = 0; k < n; ++k) {
    by_a[{c.i[k], c.a[k], c.j[k]}].add(z[k]);
    by_b[{c.j[k], c.b[k], c.i[k]}].add(z[k]);
  }
  Eigen::MatrixXd x(n, 6);
  for (Eigen::Index k = 0; k < n; ++k) {
    const BestTwo& ta = by_a.at({c.i[k], c.a[k], c.j[k]});
    const BestTwo& tb = by_b.at({c.j[k], c.b[k], c.i[k]});
    const double rival = std::min(ta.rival(z[k]), tb.rival(z[k]));
    const double margin = std::max(std::min(rival - z[k], kMarginCap), -kMarginCap);
    const bool mutual = z[k] <= ta.best && z[k] <= tb.best;
    x(k, 0) = z[k] * z[k];
    x(k, 1) = margin;
    x(k, 2) = mutual ? 1.0 : 0.0;
    x(k, 3) = c.size[k];
    x(k, 4) = c.low_score[k];
    x(k, 5) = c.geotag[k];
  }
  return x;
}

Eigen::VectorXd log_odds(const Eigen::MatrixXd& x, const Eigen::VectorXd& mean,
                         const Eigen::VectorXd& std, const Eigen::VectorXd& coef) {
  if (mean.size() != x.cols() || std.size() != x.cols() || coef.size() != x.cols() + 1) {
    throw std::invalid_argument("model size does not match the features");
  }
  Eigen::VectorXd out(x.rows());
  for (Eigen::Index r = 0; r < x.rows(); ++r) {
    double t = coef[0];
    for (Eigen::Index k = 0; k < x.cols(); ++k) t += (x(r, k) - mean[k]) / std[k] * coef[k + 1];
    out[r] = t;
  }
  return out;
}

std::vector<std::vector<Node>> greedy_join(const std::vector<int>& counts, const Candidates& c,
                                           const Eigen::VectorXd& weight) {
  check_sizes(c);
  if (weight.size() != c.i.size()) throw std::invalid_argument("one weight per candidate");
  // node index: offset of each still
  std::vector<int> offset(counts.size() + 1, 0);
  for (std::size_t s = 0; s < counts.size(); ++s) offset[s + 1] = offset[s] + counts[s];
  const int n_nodes = offset.back();
  auto node = [&](int still, int det) {
    if (still < 0 || static_cast<std::size_t>(still) >= counts.size() || det < 0 ||
        det >= counts[static_cast<std::size_t>(still)]) {
      throw std::out_of_range("candidate refers to a detection that does not exist");
    }
    return offset[static_cast<std::size_t>(still)] + det;
  };
  // pairs of positive weight between different stills, strongest first, ties in input order
  std::vector<Eigen::Index> order;
  for (Eigen::Index k = 0; k < c.i.size(); ++k) {
    if (c.i[k] != c.j[k] && weight[k] > 0) order.push_back(k);
  }
  std::stable_sort(order.begin(), order.end(),
                   [&](Eigen::Index p, Eigen::Index q) { return weight[p] > weight[q]; });
  std::vector<int> parent(static_cast<std::size_t>(n_nodes));
  std::iota(parent.begin(), parent.end(), 0);
  std::vector<std::set<int>> stills(static_cast<std::size_t>(n_nodes));
  std::vector<std::vector<int>> members(static_cast<std::size_t>(n_nodes));
  for (std::size_t s = 0; s < counts.size(); ++s) {
    for (int d = 0; d < counts[s]; ++d) {
      const auto u = static_cast<std::size_t>(offset[s] + d);
      stills[u] = {static_cast<int>(s)};
      members[u] = {static_cast<int>(u)};
    }
  }
  auto find = [&](int u) {
    while (parent[static_cast<std::size_t>(u)] != u) {
      auto& p = parent[static_cast<std::size_t>(u)];
      p = parent[static_cast<std::size_t>(p)];
      u = p;
    }
    return u;
  };
  for (const Eigen::Index k : order) {
    int ra = find(node(c.i[k], c.a[k]));
    int rb = find(node(c.j[k], c.b[k]));
    if (ra == rb) continue;
    auto& sa = stills[static_cast<std::size_t>(ra)];
    auto& sb = stills[static_cast<std::size_t>(rb)];
    const bool clash = std::any_of(sb.begin(), sb.end(), [&](int s) { return sa.count(s) > 0; });
    if (clash) continue;  // one still would appear twice
    if (members[static_cast<std::size_t>(ra)].size() <
        members[static_cast<std::size_t>(rb)].size()) {
      std::swap(ra, rb);
    }
    parent[static_cast<std::size_t>(rb)] = ra;
    auto& ma = members[static_cast<std::size_t>(ra)];
    auto& mb = members[static_cast<std::size_t>(rb)];
    ma.insert(ma.end(), mb.begin(), mb.end());
    mb.clear();
    auto& ta = stills[static_cast<std::size_t>(ra)];
    auto& tb = stills[static_cast<std::size_t>(rb)];
    ta.insert(tb.begin(), tb.end());
    tb.clear();
  }
  // node index back to (still, detection)
  std::vector<Node> name(static_cast<std::size_t>(n_nodes));
  for (std::size_t s = 0; s < counts.size(); ++s) {
    for (int d = 0; d < counts[s]; ++d) {
      name[static_cast<std::size_t>(offset[s] + d)] = {static_cast<int>(s), d};
    }
  }
  std::vector<std::vector<Node>> groups;
  for (int u = 0; u < n_nodes; ++u) {
    if (find(u) != u) continue;
    std::vector<Node> g;
    for (const int m : members[static_cast<std::size_t>(u)]) {
      g.push_back(name[static_cast<std::size_t>(m)]);
    }
    std::sort(g.begin(), g.end());
    groups.push_back(std::move(g));
  }
  std::sort(groups.begin(), groups.end());
  return groups;
}

}  // namespace herdsight::count
