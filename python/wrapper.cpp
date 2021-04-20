// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <xtensor/xadapt.hpp>

#include <basix/cell.h>
#include <basix/element-families.h>
#include <basix/finite-element.h>
#include <basix/indexing.h>
#include <basix/lattice.h>
#include <basix/maps.h>
#include <basix/polyset.h>
#include <basix/quadrature.h>
#include <xtl/xspan.hpp>

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

namespace
{
auto adapt_x(const py::array_t<double, py::array::c_style>& x)
{
  std::vector<std::size_t> shape;
  for (pybind11::ssize_t i = 0; i < x.ndim(); ++i)
    shape.push_back(x.shape(i));
  return xt::adapt(x.data(), x.size(), xt::no_ownership(), shape);
}
} // namespace

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
      .value("point", cell::type::point)
      .value("interval", cell::type::interval)
      .value("triangle", cell::type::triangle)
      .value("tetrahedron", cell::type::tetrahedron)
      .value("quadrilateral", cell::type::quadrilateral)
      .value("hexahedron", cell::type::hexahedron)
      .value("prism", cell::type::prism)
      .value("pyramid", cell::type::pyramid);

  m.def("topology", &cell::topology,
        "Topological description of a reference cell");
  m.def(
      "geometry",
      [](cell::type celltype) {
        xt::xtensor<double, 2> g = cell::geometry(celltype);
        auto strides = g.strides();
        for (auto& s : strides)
          s *= sizeof(double);
        return py::array_t<double>(g.shape(), strides, g.data());
      },
      "Geometric points of a reference cell");
  m.def(
      "sub_entity_geometry",
      [](cell::type celltype, int dim, int index) {
        xt::xtensor<double, 2> g
            = cell::sub_entity_geometry(celltype, dim, index);
        auto strides = g.strides();
        for (auto& s : strides)
          s *= sizeof(double);
        return py::array_t<double>(g.shape(), strides, g.data());
      },
      "Points of a sub-entity of a cell");

  py::enum_<lattice::type>(m, "LatticeType")
      .value("equispaced", lattice::type::equispaced)
      .value("gll_warped", lattice::type::gll_warped);

  m.def(
      "create_lattice",
      [](cell::type celltype, int n, lattice::type type, bool exterior) {
        auto l = lattice::create(celltype, n, type, exterior);
        auto strides = l.strides();
        for (auto& s : strides)
          s *= sizeof(double);
        return py::array_t<double>(l.shape(), strides, l.data());
      },
      "Create a uniform lattice of points on a reference cell");

  py::enum_<maps::type>(m, "MappingType")
      .value("identity", maps::type::identity)
      .value("covariantPiola", maps::type::covariantPiola)
      .value("contravariantPiola", maps::type::contravariantPiola)
      .value("doubleCovariantPiola", maps::type::doubleCovariantPiola)
      .value("doubleContravariantPiola", maps::type::doubleContravariantPiola);

  m.def(
      "mapping_to_str",
      [](maps::type mapping_type) -> const std::string& {
        return maps::type_to_str(mapping_type);
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
        return element::family_to_str(family_type);
      },
      "Convert a family type to a string.");

  py::enum_<element::variant>(m, "ElementVariant")
      .value("DEFAULT", element::variant::DEFAULT)
      .value("EQ", element::variant::EQ)
      .value("GLL", element::variant::GLL);

  m.def(
      "variant_to_str",
      [](element::variant variant) -> const std::string& {
        return element::variant_to_str(variant);
      },
      "Convert a element variant to a string.");

  // m.def(
  //     "create_new_element",
  //     [](element::family family_type, cell::type celltype, int degree,
  //        std::vector<std::size_t>& value_shape,
  //        const py::array_t<double, py::array::c_style>& interpolation_points,
  //        const py::array_t<double, py::array::c_style>& interpolation_matrix,
  //        const py::array_t<double, py::array::c_style>& coeffs,
  //        const std::vector<std::vector<int>>& entity_dofs,
  //        const py::array_t<double, py::array::c_style>& base_transformations,
  //        maps::type mapping_type = maps::type::identity) -> FiniteElement {
  //       return FiniteElement(family_type, celltype, degree, value_shape,
  //                            compute_expansion_coefficients(
  //                                celltype, adapt_x(coeffs),
  //                                adapt_x(interpolation_matrix),
  //                                adapt_x(interpolation_points),
  //                                degree, 1.0e6),
  //                            entity_dofs, adapt_x(base_transformations),
  //                            adapt_x(interpolation_points),
  //                            adapt_x(interpolation_matrix), mapping_type);
  //     },
  //     "Create an element from basic data");

  // m.def(
  //     "create_new_element",
  //     [](std::string family_name, std::string cell_name, int degree,
  //        std::vector<std::size_t>& value_shape,
  //        const py::array_t<double, py::array::c_style>& interpolation_points,
  //        const py::array_t<double, py::array::c_style>& interpolation_matrix,
  //        const py::array_t<double, py::array::c_style>& coeffs,
  //        const std::vector<std::vector<int>>& entity_dofs,
  //        const py::array_t<double, py::array::c_style>& base_transformations,
  //        maps::type mapping_type = maps::type::identity) -> FiniteElement {
  //       return FiniteElement(element::str_to_type(family_name),
  //                            cell::str_to_type(cell_name), degree,
  //                            value_shape, compute_expansion_coefficients(
  //                                cell::str_to_type(cell_name),
  //                                adapt_x(coeffs),
  //                                adapt_x(interpolation_matrix),
  //                                adapt_x(interpolation_points),
  //                                degree, 1.0e6),
  //                            entity_dofs, adapt_x(base_transformations),
  //                            adapt_x(interpolation_points),
  //                            adapt_x(interpolation_matrix), mapping_type);
  //     },
  //     "Create an element from basic data");

  py::class_<FiniteElement>(m, "FiniteElement", "Finite Element")
      .def(
          "tabulate",
          [](const FiniteElement& self, int n,
             const py::array_t<double, py::array::c_style>& x) {
            auto _x = adapt_x(x);
            auto t = self.tabulate(n, _x);
            auto t_swap = xt::transpose(t, {0, 1, 3, 2});
            xt::xtensor<double, 3> t_reshape
                = xt::reshape_view(t_swap, {t_swap.shape(0), t_swap.shape(1),
                                            t_swap.shape(2) * t_swap.shape(3)});
            return py::array_t<double>(t_reshape.shape(), t_reshape.data());
          },
          tabdoc.c_str())
      .def(
          "tabulate_x",
          [](const FiniteElement& self, int n,
             const py::array_t<double, py::array::c_style>& x) {
            auto _x = adapt_x(x);
            auto t = self.tabulate(n, _x);
            return py::array_t<double>(t.shape(), t.data());
          },
          tabdoc.c_str())
      .def(
          "map_push_forward",
          [](const FiniteElement& self,
             const py::array_t<double, py::array::c_style>& U,
             const py::array_t<double, py::array::c_style>& J,
             const py::array_t<double, py::array::c_style>& detJ,
             const py::array_t<double, py::array::c_style>& K) {
            auto u = self.map_push_forward(
                adapt_x(U), adapt_x(J),
                xtl::span<const double>(detJ.data(), detJ.size()), adapt_x(K));
            return py::array_t<double>(u.shape(), u.data());
          },
          mapdoc.c_str())
      .def(
          "map_pull_back",
          [](const FiniteElement& self,
             const py::array_t<double, py::array::c_style>& u,
             const py::array_t<double, py::array::c_style>& J,
             const py::array_t<double, py::array::c_style>& detJ,
             const py::array_t<double, py::array::c_style>& K) {
            auto U = self.map_pull_back(
                adapt_x(u), adapt_x(J),
                xtl::span<const double>(detJ.data(), detJ.size()), adapt_x(K));
            return py::array_t<double>(U.shape(), U.data());
          },
          invmapdoc.c_str())
      .def("base_transformations",
           [](const FiniteElement& self) {
             xt::xtensor<double, 3> t = self.base_transformations();
             return py::array_t<double>(t.shape(), t.data());
           })
      .def_property_readonly("degree", &FiniteElement::degree)
      .def_property_readonly("cell_type", &FiniteElement::cell_type)
      .def_property_readonly("dim", &FiniteElement::dim)
      .def_property_readonly("entity_dofs", &FiniteElement::entity_dofs)
      .def_property_readonly("value_size", &FiniteElement::value_size)
      .def_property_readonly("value_shape", &FiniteElement::value_shape)
      .def_property_readonly("family", &FiniteElement::family)
      .def_property_readonly(
          "dof_transformations_are_permutations",
          &FiniteElement::dof_transformations_are_permutations)
      .def_property_readonly("dof_transformations_are_identity",
                             &FiniteElement::dof_transformations_are_identity)
      .def_property_readonly("mapping_type", &FiniteElement::mapping_type)
      .def_property_readonly("points",
                             [](const FiniteElement& self) {
                               const xt::xtensor<double, 2>& x = self.points();
                               return py::array_t<double>(x.shape(), x.data(),
                                                          py::cast(self));
                             })
      .def_property_readonly(
          "interpolation_matrix", [](const FiniteElement& self) {
            const xt::xtensor<double, 2>& P = self.interpolation_matrix();
            return py::array_t<double>(P.shape(), P.data(), py::cast(self));
          });

  // Create FiniteElement
  m.def(
      "create_element",
      [](const std::string family_name, const std::string cell_name,
         int degree) -> FiniteElement {
        return basix::create_element(family_name, cell_name, degree);
      },
      "Create a FiniteElement of a given family, celltype and degree");

  m.def(
      "create_element",
      [](const std::string family_name, const std::string cell_name, int degree,
         const std::string variant) -> FiniteElement {
        return basix::create_element(family_name, cell_name, degree, variant);
      },
      "Create a FiniteElement of a given family, celltype, degree and variant");

  m.def(
      "tabulate_polynomial_set",
      [](cell::type celltype, int d, int n,
         const py::array_t<double, py::array::c_style>& x) {
        std::vector<std::size_t> shape;
        if (x.ndim() == 2 and x.shape(1) == 1)
          shape.push_back(x.shape(0));
        else
        {
          for (pybind11::ssize_t i = 0; i < x.ndim(); ++i)
            shape.push_back(x.shape(i));
        }
        auto _x = xt::adapt(x.data(), x.size(), xt::no_ownership(), shape);
        xt::xtensor<double, 3> P = polyset::tabulate(celltype, d, n, _x);
        return py::array_t<double>(P.shape(), P.data());
      },
      "Tabulate orthonormal polynomial expansion set");

  m.def(
      "compute_jacobi_deriv",
      [](double a, std::size_t n, std::size_t nderiv,
         const py::array_t<double, py::array::c_style>& x) {
        if (x.ndim() > 1)
          throw std::runtime_error("Expected 1D x array.");
        xt::xtensor<double, 2> f = quadrature::compute_jacobi_deriv(
            a, n, nderiv, xtl::span<const double>(x.data(), x.size()));
        return py::array_t<double>(f.shape(), f.data());
      },
      "Compute jacobi polynomial and derivatives at points");

  m.def(
      "make_quadrature",
      [](const std::string& rule, cell::type celltype, int m) {
        auto [pts, w] = quadrature::make_quadrature(rule, celltype, m);
        // FIXME: it would be more elegant to handle 1D case as a 1D
        // array, but FFC would need updating
        if (pts.dimension() == 1)
        {
          std::array<std::size_t, 2> s = {pts.shape(0), 1};
          return std::pair(py::array_t<double>(s, pts.data()),
                           py::array_t<double>(w.size(), w.data()));
        }
        else
        {
          return std::pair(py::array_t<double>(pts.shape(), pts.data()),
                           py::array_t<double>(w.size(), w.data()));
        }
      },
      "Compute quadrature points and weights on a reference cell");

  m.def("index", py::overload_cast<int>(&basix::idx), "Indexing for 1D arrays")
      .def("index", py::overload_cast<int, int>(&basix::idx),
           "Indexing for triangular arrays")
      .def("index", py::overload_cast<int, int, int>(&basix::idx),
           "Indexing for tetrahedral arrays");
}
