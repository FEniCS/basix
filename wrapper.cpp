#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "lagrange.h"
#include "nedelec.h"
#include "quadrature.h"
#include "raviart-thomas.h"
#include "simplex.h"

namespace py = pybind11;

PYBIND11_MODULE(fiatx, m)
{
  m.doc() = "FIATx example plugin";

  m.def("create_lattice", &ReferenceSimplex::create_lattice,
        "Create a lattice");

  py::class_<Nedelec2D>(m, "Nedelec2D")
      .def(py::init<int>())
      .def("tabulate_basis", &Nedelec2D::tabulate_basis);

  py::class_<Lagrange>(m, "Lagrange")
      .def(py::init<int, int>())
      .def("tabulate_basis", &Lagrange::tabulate_basis);

  py::class_<RaviartThomas>(m, "RaviartThomas")
      .def(py::init<int, int>())
      .def("tabulate_basis", &RaviartThomas::tabulate_basis);

  m.def("make_quadrature",
        py::overload_cast<const Eigen::Array<double, Eigen::Dynamic,
                                             Eigen::Dynamic, Eigen::RowMajor>&,
                          int>(&make_quadrature));
}
