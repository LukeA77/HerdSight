// Python bindings for the HerdSight C++ core: import as herdsight._core.
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "herdsight/frames.hpp"

namespace py = pybind11;
namespace fr = herdsight::frames;

PYBIND11_MODULE(_core, m) {
  m.doc() = "HerdSight C++ core (geometry). Python references live in herdsight.geo.";
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
}
