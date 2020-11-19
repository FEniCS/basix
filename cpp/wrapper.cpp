// Copyright (c) 2020 Chris Richardson and Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "cell.h"
#include "indexing.h"
#include "lattice.h"
#include "libtab.h"
#include "polyset.h"
#include "quadrature.h"

// TODO: remove, not in public interface
#include "crouzeix-raviart.h"
#include "lagrange.h"
#include "nedelec.h"
#include "raviart-thomas.h"
#include "regge.h"

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

  py::enum_<cell::type>(m, "CellType")
      .value("interval", cell::type::interval)
      .value("triangle", cell::type::triangle)
      .value("tetrahedron", cell::type::tetrahedron)
      .value("quadrilateral", cell::type::quadrilateral)
      .value("hexahedron", cell::type::hexahedron)
      .value("prism", cell::type::prism)
      .value("pyramid", cell::type::pyramid);

  m.def("topology", &cell::topology,
        "Topological description of a reference cell");
  m.def("geometry", &cell::geometry, "Geometric points of a reference cell");
  m.def("sub_entity_geometry", &cell::sub_entity_geometry,
        "Points of a sub-entity of a cell");

  py::enum_<lattice::type>(m, "LatticeType")
      .value("equispaced", lattice::type::equispaced)
      .value("gll_warped", lattice::type::gll_warped);

  m.def("create_lattice", &lattice::create,
        "Create a uniform lattice of points on a reference cell");

  m.def(
      "create_new_element",
      [](const std::string family_name, cell::type celltype, int degree,
         std::vector<int>& value_shape, const Eigen::MatrixXd& dualmat,
         const Eigen::MatrixXd& coeffs,
         const std::vector<std::vector<int>>& entity_dofs,
         const std::vector<Eigen::MatrixXd>& base_permutations)
          -> FiniteElement {
        Eigen::MatrixXd new_coeffs
            = FiniteElement::compute_expansion_coefficients(coeffs, dualmat,
                                                            true);
        return FiniteElement(family_name, celltype, degree, value_shape,
                             new_coeffs, entity_dofs, base_permutations);
      },
      "Create an element from basic data");

  py::class_<FiniteElement>(m, "FiniteElement", "Finite Element")
      .def("tabulate", &FiniteElement::tabulate, tabdoc.c_str())
      .def_property_readonly("base_permutations",
                             &FiniteElement::base_permutations)
      .def_property_readonly("degree", &FiniteElement::degree)
      .def_property_readonly("cell_type", &FiniteElement::cell_type)
      .def_property_readonly("ndofs", &FiniteElement::dim)
      .def_property_readonly("entity_dofs", &FiniteElement::entity_dofs)
      .def_property_readonly("value_size", &FiniteElement::value_size)
      .def_property_readonly("value_shape", &FiniteElement::value_shape)
      .def_property_readonly("family_name", &FiniteElement::family_name);

  // TODO: remove - not part of public interface
  // Create FiniteElement of different types
  m.def("Nedelec", [](const std::string& cell, int degree) {
    return libtab::create_element("Nedelec 1st kind H(curl)", cell, degree);
  });
  m.def("NedelecSecondKind", [](const std::string& cell, int degree) {
    return libtab::create_element("Nedelec 2nd kind H(curl)", cell, degree);
  });
  m.def("Lagrange", [](const std::string& cell, int degree) {
    return libtab::create_element("Lagrange", cell, degree);
  });
  m.def("DiscontinuousLagrange", [](const std::string& cell, int degree) {
    return libtab::create_element("Discontinuous Lagrange", cell, degree);
  });
  m.def("CrouzeixRaviart", [](const std::string& cell, int degree) {
    return libtab::create_element("Crouzeix-Raviart", cell, degree);
  });
  m.def("RaviartThomas", [](const std::string& cell, int degree) {
    return libtab::create_element("Raviart-Thomas", cell, degree);
  });
  m.def("Regge", [](const std::string& cell, int degree) {
    return libtab::create_element("Regge", cell, degree);
  });

  // Create FiniteElement
  m.def("create_element", &libtab::create_element,
        "Create a FiniteElement of a given family, celltype and degree");

  m.def("tabulate_polynomial_set", &polyset::tabulate,
        "Tabulate orthonormal polynomial expansion set");

  m.def("compute_jacobi_deriv", &quadrature::compute_jacobi_deriv,
        "Compute jacobi polynomial and derivatives at points");

  m.def("make_quadrature",
        py::overload_cast<const Eigen::ArrayXXd&, int>(
            &quadrature::make_quadrature),
        "Compute quadrature points and weights on a simplex defined by points")
      .def("make_quadrature",
           py::overload_cast<cell::type, int>(&quadrature::make_quadrature),
           "Compute quadrature points and weights on a reference cell");

  m.def("gauss_lobatto_legendre_line_rule",
        &quadrature::gauss_lobatto_legendre_line_rule,
        "Compute GLL quadrature points and weights on the interval [-1, 1]");

  m.def("index", py::overload_cast<int>(&libtab::idx), "Indexing for 1D arrays")
      .def("index", py::overload_cast<int, int>(&libtab::idx),
           "Indexing for triangular arrays")
      .def("index", py::overload_cast<int, int, int>(&libtab::idx),
           "Indexing for tetrahedral arrays");
}
