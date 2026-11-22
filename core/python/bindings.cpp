// Python bindings for the HerdSight C++ core: import as herdsight._core.
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "herdsight/count.hpp"
#include "herdsight/frames.hpp"

namespace py = pybind11;
namespace fr = herdsight::frames;
namespace ct = herdsight::count;

PYBIND11_MODULE(_core, m) {
  m.doc() = "HerdSight C++ core. Python references live in herdsight.geo and herdsight.count.";
  m.def("rot_x", &fr::rot_x, py::arg("a"));
  m.def("rot_y", &fr::rot_y, py::arg("a"));
  m.def("rot_z", &fr::rot_z, py::arg("a"));
  m.def("R_wb_from_ypr", &fr::R_wb_from_ypr, py::arg("yaw"), py::arg("pitch"), py::arg("roll"));
  m.def(
      "ypr_from_R_wb",
      [](const Eigen::Matrix3d& R) {
        const auto v = fr::ypr_from_R_wb(R);
        return py::make_tuple(v.yaw, v.pitch, v.roll);
      },
      py::arg("R_wb"));

  auto candidates = [](const Eigen::VectorXi& i, const Eigen::VectorXi& a, const Eigen::VectorXi& j,
                       const Eigen::VectorXi& b, const Eigen::VectorXd& d_m,
                       const Eigen::VectorXd& dt_s, const Eigen::VectorXd& geotag,
                       const Eigen::VectorXd& fixed_gate_m, const Eigen::VectorXd& size,
                       const Eigen::VectorXd& low_score) {
    return ct::Candidates{i, a, j, b, d_m, dt_s, geotag, fixed_gate_m, size, low_score};
  };
  m.def(
      "pair_features",
      [candidates](const Eigen::VectorXi& i, const Eigen::VectorXi& a, const Eigen::VectorXi& j,
                   const Eigen::VectorXi& b, const Eigen::VectorXd& d_m,
                   const Eigen::VectorXd& dt_s, const Eigen::VectorXd& geotag,
                   const Eigen::VectorXd& fixed_gate_m, const Eigen::VectorXd& size,
                   const Eigen::VectorXd& low_score, double diffusion_m2_s,
                   const std::string& kind) {
        if (kind != "fixed" && kind != "movement") throw py::value_error(kind);
        return ct::features(
            candidates(i, a, j, b, d_m, dt_s, geotag, fixed_gate_m, size, low_score),
            diffusion_m2_s, kind == "fixed" ? ct::Scale::Fixed : ct::Scale::Movement);
      },
      py::arg("i"), py::arg("a"), py::arg("j"), py::arg("b"), py::arg("d_m"), py::arg("dt_s"),
      py::arg("geotag"), py::arg("fixed_gate_m"), py::arg("size"), py::arg("low_score"),
      py::arg("diffusion_m2_s"), py::arg("kind"));
  m.def("log_odds", &ct::log_odds, py::arg("x"), py::arg("mean"), py::arg("std"),
        py::arg("coef"));
  m.def(
      "greedy_join",
      [candidates](const std::vector<int>& counts, const Eigen::VectorXi& i,
                   const Eigen::VectorXi& a, const Eigen::VectorXi& j, const Eigen::VectorXi& b,
                   const Eigen::VectorXd& weight) {
        const auto n = i.size();
        const Eigen::VectorXd zero = Eigen::VectorXd::Zero(n);
        return ct::greedy_join(counts,
                               candidates(i, a, j, b, zero, zero, zero, zero, zero, zero),
                               weight);
      },
      py::arg("counts"), py::arg("i"), py::arg("a"), py::arg("j"), py::arg("b"),
      py::arg("weight"));
}
