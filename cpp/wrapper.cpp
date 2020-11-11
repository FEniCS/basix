// Copyright (c) 2020 Chris Richardson and Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "cell.h"
#include "crouzeix-raviart.h"
#include "defines.h"
#include "indexing.h"
#include "lagrange.h"
#include "nedelec-second-kind.h"
#include "nedelec.h"
#include "polynomial-set.h"
#include "quadrature.h"
#include "raviart-thomas.h"
#include "regge.h"
#include "tp.h"

namespace py = pybind11;
using namespace libtab;

const std::string tabdoc = R"(
Tabulate the finite element basis function and derivatives at points.
If no derivatives are required, use nderiv=0. In 2D and 3D, the derivatives are ordered
in triangular (or tetrahedral) order, i.e. (0,0),(1,0),(0,1),(2,0),(1,1),(0,2) etc. in 2D.

Parameters
==========
nderiv : int
    Number of derivatives required

points : numpy.ndarray
    Array of points

Returns
=======
List[numpy.ndarray]
    List of basis values and derivatives at points. Returns a list of length `(n+d)!/(n!d!)`
    where `n` is the number of derivatives and `d` is the topological dimension.
)";

PYBIND11_MODULE(libtab, m)
{
  m.doc() = R"(
Libtab provides information about finite elements on the reference cell. It has support for
interval (1D), triangle and quadrilateral (2D), and tetrahedron, hexahedron, prism and pyramid (3D) reference cells.
Elements are available in several different types, typically as `ElementName(celltype, degree)`. Not all elements are available
on all cell types, and an error should be thrown if an invalid combination is requested.
Each element has a `tabulate` function which returns the basis functions and a number of their derivatives, as desired.

)";

  m.attr("__version__") = libtab::version();

  py::enum_<cell::Type>(m, "CellType")
      .value("interval", cell::Type::interval)
      .value("triangle", cell::Type::triangle)
      .value("tetrahedron", cell::Type::tetrahedron)
      .value("quadrilateral", cell::Type::quadrilateral)
      .value("hexahedron", cell::Type::hexahedron)
      .value("prism", cell::Type::prism)
      .value("pyramid", cell::Type::pyramid);

  m.def("topology", &cell::topology,
        "Topological description of a reference cell");
  m.def("geometry", &cell::geometry, "Geometric points of a reference cell");
  m.def("sub_entity_geometry", &cell::sub_entity_geometry,
        "Points of a sub-entity of a cell");

  m.def("simplex_type", &cell::simplex_type,
        "Simplex CellType of given dimension");
  m.def("create_lattice", &cell::create_lattice,
        "Create a uniform lattice of points on a reference cell");

  m.def(
      "create_new_element",
      [](cell::Type celltype, int degree, const std::vector<int>& value_shape,
         const Eigen::MatrixXd& dualmat, const Eigen::MatrixXd& coeffs,
         const std::vector<std::vector<int>>& entity_dofs) -> FiniteElement {
        auto new_coeffs = FiniteElement::compute_expansion_coefficents(
            coeffs, dualmat, true);
        return FiniteElement(celltype, degree, value_shape, new_coeffs,
                             entity_dofs);
      },
      "Create an element from basic data");

  py::class_<FiniteElement>(m, "FiniteElement", "Finite Element")
      .def("tabulate", &FiniteElement::tabulate, tabdoc.c_str())
      .def_property_readonly("degree", &FiniteElement::degree)
      .def_property_readonly("cell_type", &FiniteElement::cell_type)
      .def_property_readonly("ndofs", &FiniteElement::ndofs)
      .def_property_readonly("entity_dofs", &FiniteElement::entity_dofs)
      .def_property_readonly("value_size", &FiniteElement::value_size);

  // Create FiniteElement of different types
  m.def("Nedelec", &Nedelec::create, "Create Nedelec Element (first kind)");
  m.def("Lagrange", &Lagrange::create, "Create Lagrange Element");
  m.def("CrouzeixRaviart", &CrouzeixRaviart::create,
        "Create Crouzeix-Raviart Element");
  m.def("TensorProduct", &TensorProduct::create,
        "Create TensorProduct Element");
  m.def("RaviartThomas", &RaviartThomas::create,
        "Create Raviart-Thomas Element");
  m.def("NedelecSecondKind", &NedelecSecondKind::create,
        "Create Nedelec Element (second kind)");
  m.def("Regge", &Regge::create, "Create Regge Element");

  m.def(
      "create_element",
      [](std::string family, std::string cell, int degree) {
        const std::map<std::string,
                       std::function<FiniteElement(cell::Type, int)>>
            create_map
            = {{"Crouzeix-Raviart", &CrouzeixRaviart::create},
               {"Discontinuous Lagrange", &Lagrange::create},
               {"Lagrange", &Lagrange::create},
               {"Nedelec 1st kind H(curl)", &Nedelec::create},
               {"Nedelec 2nd kind H(curl)", &NedelecSecondKind::create},
               {"Raviart-Thomas", &RaviartThomas::create},
               {"Regge", &Regge::create}};

        auto create_it = create_map.find(family);
        if (create_it == create_map.end())
          throw std::runtime_error("Family not found: \"" + family + "\"");

        const cell::Type celltype = cell::str_to_type(cell);
        return create_it->second(celltype, degree);
      },
      "Create a FiniteElement of a given family, celltype and degree");

  m.def("tabulate_polynomial_set", &polyset::tabulate,
        "Tabulate orthonormal polynomial expansion set");

  m.def("compute_jacobi_deriv", &quadrature::compute_jacobi_deriv,
        "Compute jacobi polynomial and derivatives at points");

  m.def("make_quadrature",
        py::overload_cast<const Eigen::Array<double, Eigen::Dynamic,
                                             Eigen::Dynamic, Eigen::RowMajor>&,
                          int>(&quadrature::make_quadrature),
        "Compute quadrature points and weights on a simplex defined by points");

  m.def("gauss_lobatto_legendre_line_rule",
        &quadrature::gauss_lobatto_legendre_line_rule,
        "Compute GLL quadrature points and weights on the interval [-1, 1]");

  m.def("index", py::overload_cast<int>(&libtab::idx), "Indexing for 1D arrays")
      .def("index", py::overload_cast<int, int>(&libtab::idx),
           "Indexing for triangular arrays")
      .def("index", py::overload_cast<int, int, int>(&libtab::idx),
           "Indexing for tetrahedral arrays");
}
