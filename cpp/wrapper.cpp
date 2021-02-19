// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

#include "cell.h"
#include "element-families.h"
#include "finite-element.h"
#include "indexing.h"
#include "lattice.h"
#include "mappings.h"
#include "polyset.h"
#include "quadrature.h"

// TODO: remove, not in public interface
#include "crouzeix-raviart.h"
#include "lagrange.h"
#include "nedelec.h"
#include "raviart-thomas.h"
#include "regge.h"

namespace py = pybind11;
using namespace basix;

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

const std::string mapdoc = R"(
Map a function value from the reference to a cell

Parameters
==========
reference_data : numpy.array
    The function value on the reference
J : np.ndarray
    The Jacobian of the mapping
detJ : int
    The determinant of the Jacobian of the mapping
K : np.ndarray
    The inverse of the Jacobian of the mapping

Returns
=======
numpy.ndarray
    The function value on the cell
)";

const std::string invmapdoc = R"(
Map a function value from a cell to the reference

Parameters
==========
physical_data : numpy.array
    The function value on the cell
J : np.ndarray
    The Jacobian of the mapping
detJ : int
    The determinant of the Jacobian of the mapping
K : np.ndarray
    The inverse of the Jacobian of the mapping

Returns
=======
numpy.ndarray
    The function value on the reference
)";

PYBIND11_MODULE(_basixcpp, m)
{
  m.doc() = R"(
basix provides information about finite elements on the reference cell. It has support for
interval (1D), triangle and quadrilateral (2D), and tetrahedron, hexahedron, prism and pyramid (3D) reference cells.
Elements are available in several different types, typically as `ElementName(celltype, degree)`. Not all elements are available
on all cell types, and an error should be thrown if an invalid combination is requested.
Each element has a `tabulate` function which returns the basis functions and a number of their derivatives, as desired.

)";

  m.attr("__version__") = basix::version();

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

  py::enum_<mapping::type>(m, "MappingType")
      .value("identity", mapping::type::identity)
      .value("covariantPiola", mapping::type::covariantPiola)
      .value("contravariantPiola", mapping::type::contravariantPiola)
      .value("doubleCovariantPiola", mapping::type::doubleCovariantPiola)
      .value("doubleContravariantPiola",
             mapping::type::doubleContravariantPiola);

  m.def(
      "mapping_to_str",
      [](mapping::type mapping_type) -> const std::string& {
        return mapping::type_to_str(mapping_type);
      },
      "Convert a mapping type to a string.");

  py::enum_<element::family>(m, "ElementFamily")
      .value("custom", element::family::custom)
      .value("P", element::family::P)
      .value("DP", element::family::DP)
      .value("BDM", element::family::BDM)
      .value("RT", element::family::RT)
      .value("N1E", element::family::N1E)
      .value("N2E", element::family::N2E)
      .value("Regge", element::family::Regge)
      .value("CR", element::family::CR);

  m.def(
      "family_to_str",
      [](element::family family_type) -> const std::string& {
        return element::type_to_str(family_type);
      },
      "Convert a family type to a string.");

  m.def(
      "create_new_element",
      [](element::family family_type, cell::type celltype, int degree,
         std::vector<int>& value_shape, const Eigen::MatrixXd& dualmat,
         const Eigen::MatrixXd& coeffs,
         const std::vector<std::vector<int>>& entity_dofs,
         const std::vector<Eigen::MatrixXd>& base_permutations,
         mapping::type mapping_type
         = mapping::type::identity) -> FiniteElement {
        return FiniteElement(
            family_type, celltype, degree, value_shape,
            compute_expansion_coefficients_legacy(coeffs, dualmat, true),
            entity_dofs, base_permutations, {}, {}, mapping_type);
      },
      "Create an element from basic data");

  m.def(
      "create_new_element",
      [](std::string family_name, std::string cell_name, int degree,
         std::vector<int>& value_shape, const Eigen::MatrixXd& dualmat,
         const Eigen::MatrixXd& coeffs,
         const std::vector<std::vector<int>>& entity_dofs,
         const std::vector<Eigen::MatrixXd>& base_permutations,
         mapping::type mapping_type
         = mapping::type::identity) -> FiniteElement {
        return FiniteElement(
            element::str_to_type(family_name), cell::str_to_type(cell_name),
            degree, value_shape,
            compute_expansion_coefficients_legacy(coeffs, dualmat, true),
            entity_dofs, base_permutations, {}, {}, mapping_type);
      },
      "Create an element from basic data");

  py::class_<FiniteElement>(m, "FiniteElement", "Finite Element")
      .def("tabulate", &FiniteElement::tabulate, tabdoc.c_str())
      .def("map_push_forward", &FiniteElement::map_push_forward, mapdoc.c_str())
      .def("map_pull_back", &FiniteElement::map_pull_back, invmapdoc.c_str())
      .def_property_readonly("base_permutations",
                             &FiniteElement::base_permutations)
      .def_property_readonly("degree", &FiniteElement::degree)
      .def_property_readonly("cell_type", &FiniteElement::cell_type)
      .def_property_readonly("dim", &FiniteElement::dim)
      .def_property_readonly("entity_dofs", &FiniteElement::entity_dofs)
      .def_property_readonly("value_size", &FiniteElement::value_size)
      .def_property_readonly("value_shape", &FiniteElement::value_shape)
      .def_property_readonly("family", &FiniteElement::family)
      .def_property_readonly("mapping_type", &FiniteElement::mapping_type)
      .def_property_readonly("points", &FiniteElement::points)
      .def_property_readonly("interpolation_matrix",
                             &FiniteElement::interpolation_matrix);

  // Create FiniteElement
  m.def(
      "create_element",
      [](const std::string family_name, const std::string cell_name,
         int degree) -> FiniteElement {
        return basix::create_element(family_name, cell_name, degree);
      },
      "Create a FiniteElement of a given family, celltype and degree");

  m.def("tabulate_polynomial_set", &polyset::tabulate,
        "Tabulate orthonormal polynomial expansion set");

  m.def("compute_jacobi_deriv", &quadrature::compute_jacobi_deriv,
        "Compute jacobi polynomial and derivatives at points");

  m.def("make_quadrature", &quadrature::make_quadrature,
        "Compute quadrature points and weights on a reference cell");

  m.def("gauss_lobatto_legendre_line_rule",
        &quadrature::gauss_lobatto_legendre_line_rule,
        "Compute GLL quadrature points and weights on the interval [-1, 1]");

  m.def("index", py::overload_cast<int>(&basix::idx), "Indexing for 1D arrays")
      .def("index", py::overload_cast<int, int>(&basix::idx),
           "Indexing for triangular arrays")
      .def("index", py::overload_cast<int, int, int>(&basix::idx),
           "Indexing for tetrahedral arrays");
}
