#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "cell.h"
#include "lagrange.h"
#include "nedelec.h"
#include "quadrature.h"
#include "raviart-thomas.h"
#include "simplex.h"

namespace py = pybind11;

PYBIND11_MODULE(fiatx, m)
{
  m.doc() = "FIATx/libtab plugin";

  py::enum_<CellType>(m, "CellType")
      .value("interval", CellType::interval)
      .value("triangle", CellType::triangle)
      .value("tetrahedron", CellType::tetrahedron)
      .value("quadrilateral", CellType::quadrilateral)
      .value("hexahedron", CellType::hexahedron)
      .value("prism", CellType::prism)
      .value("pyramid", CellType::pyramid);

  m.def("create_lattice", &ReferenceSimplex::create_lattice,
        "Create a lattice");

  py::class_<Nedelec2D>(m, "Nedelec2D")
      .def(py::init<int>())
      .def("tabulate_basis", &Nedelec2D::tabulate_basis);

  py::class_<Nedelec3D>(m, "Nedelec3D")
      .def(py::init<int>())
      .def("tabulate_basis", &Nedelec3D::tabulate_basis);

  py::class_<Lagrange>(m, "Lagrange")
      .def(py::init<CellType, int>())
      .def("tabulate_basis", &Lagrange::tabulate_basis);

  py::class_<RaviartThomas>(m, "RaviartThomas")
      .def(py::init<int, int>())
      .def("tabulate_basis", &RaviartThomas::tabulate_basis);

  py::class_<Polynomial>(m, "Polynomial")
      .def(py::init<>())
      .def(py::self + py::self)
      .def(py::self += py::self)
      .def(py::self - py::self)
      .def(py::self * py::self)
      .def(py::self * float())
      .def(py::self *= float())
      .def_static("zero", &Polynomial::zero)
      .def_static("one", &Polynomial::one)
      .def_static("x", &Polynomial::x)
      .def_static("y", &Polynomial::y)
      .def_static("z", &Polynomial::z)
      .def("diff", &Polynomial::diff)
      .def("tabulate",
           py::overload_cast<const Eigen::Array<
               double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&>(
               &Polynomial::tabulate, py::const_));

  m.def("make_quadrature",
        py::overload_cast<const Eigen::Array<double, Eigen::Dynamic,
                                             Eigen::Dynamic, Eigen::RowMajor>&,
                          int>(&make_quadrature));
}
