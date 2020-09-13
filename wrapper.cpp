#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "lagrange.h"
#include "nedelec.h"
#include "simplex.h"

namespace py = pybind11;

PYBIND11_MODULE(fiatx, m)
{
  m.doc() = "FIATx example plugin";

  m.def("make_lattice", &ReferenceSimplex::make_lattice, "Create a lattice");

  py::class_<Nedelec2D>(m, "Nedelec2D")
      .def(py::init<int>())
      .def("tabulate_basis", &Nedelec2D::tabulate_basis);

  py::class_<Lagrange>(m, "Lagrange")
      .def(py::init<int, int>())
      .def("tabulate_basis", &Lagrange::tabulate_basis);
}
