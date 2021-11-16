// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "docs.h"
#include <basix/cell.h>
#include <basix/element-families.h>
#include <basix/finite-element.h>
#include <basix/indexing.h>
#include <basix/interpolation.h>
#include <basix/lattice.h>
#include <basix/maps.h>
#include <basix/polyset.h>
#include <basix/quadrature.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <xtensor/xadapt.hpp>
#include <xtl/xspan.hpp>

namespace py = pybind11;
using namespace basix;

namespace
{
const std::string& cell_type_to_str(cell::type type)
{
  static const std::map<cell::type, std::string> type_to_name
      = {{cell::type::point, "point"},
         {cell::type::interval, "interval"},
         {cell::type::triangle, "triangle"},
         {cell::type::tetrahedron, "tetrahedron"},
         {cell::type::quadrilateral, "quadrilateral"},
         {cell::type::pyramid, "pyramid"},
         {cell::type::prism, "prism"},
         {cell::type::hexahedron, "hexahedron"}};

  auto it = type_to_name.find(type);
  if (it == type_to_name.end())
    throw std::runtime_error("Can't find type");

  return it->second;
}

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
Interface to the Basix C++ library.
)";

  m.attr("__version__") = basix::version();

  m.def("topology", &cell::topology, basix::docstring::topology.c_str());
  m.def(
      "geometry",
      [](cell::type celltype) {
        xt::xtensor<double, 2> g = cell::geometry(celltype);
        auto strides = g.strides();
        for (auto& s : strides)
          s *= sizeof(double);
        return py::array_t<double>(g.shape(), strides, g.data());
      },
      basix::docstring::geometry.c_str());
  m.def("sub_entity_connectivity", &cell::sub_entity_connectivity,
        basix::docstring::sub_entity_connectivity.c_str());
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
      basix::docstring::sub_entity_geometry.c_str());

  py::enum_<lattice::type>(m, "LatticeType")
      .value("equispaced", lattice::type::equispaced)
      .value("gll", lattice::type::gll)
      .value("chebyshev", lattice::type::chebyshev)
      .value("gl", lattice::type::gl);

  py::enum_<lattice::simplex_method>(m, "LatticeSimplexMethod")
      .value("none", lattice::simplex_method::none)
      .value("warp", lattice::simplex_method::warp)
      .value("isaac", lattice::simplex_method::isaac)
      .value("centroid", lattice::simplex_method::centroid);

  m.def(
      "create_lattice",
      [](cell::type celltype, int n, lattice::type type, bool exterior) {
        auto l = lattice::create(celltype, n, type, exterior,
                                 lattice::simplex_method::none);
        auto strides = l.strides();
        for (auto& s : strides)
          s *= sizeof(double);
        return py::array_t<double>(l.shape(), strides, l.data());
      },
      basix::docstring::create_lattice__celltype_n_type_exterior.c_str());

  m.def(
      "create_lattice",
      [](cell::type celltype, int n, lattice::type type, bool exterior,
         lattice::simplex_method method) {
        auto l = lattice::create(celltype, n, type, exterior, method);
        auto strides = l.strides();
        for (auto& s : strides)
          s *= sizeof(double);
        return py::array_t<double>(l.shape(), strides, l.data());
      },
      basix::docstring::create_lattice__celltype_n_type_exterior_method
          .c_str());

  py::enum_<maps::type>(m, "MappingType")
      .value("identity", maps::type::identity)
      .value("covariantPiola", maps::type::covariantPiola)
      .value("contravariantPiola", maps::type::contravariantPiola)
      .value("doubleCovariantPiola", maps::type::doubleCovariantPiola)
      .value("doubleContravariantPiola", maps::type::doubleContravariantPiola);

  py::enum_<quadrature::type>(m, "QuadratureType")
      .value("Default", quadrature::type::Default)
      .value("gauss_jacobi", quadrature::type::gauss_jacobi)
      .value("gll", quadrature::type::gll)
      .value("xiao_gimbutas", quadrature::type::xiao_gimbutas);

  py::enum_<cell::type>(m, "CellType")
      .value("point", cell::type::point)
      .value("interval", cell::type::interval)
      .value("triangle", cell::type::triangle)
      .value("tetrahedron", cell::type::tetrahedron)
      .value("quadrilateral", cell::type::quadrilateral)
      .value("hexahedron", cell::type::hexahedron)
      .value("prism", cell::type::prism)
      .value("pyramid", cell::type::pyramid);

  m.def(
      "cell_volume",
      [](cell::type cell_type) -> double { return cell::volume(cell_type); },
      basix::docstring::cell_volume.c_str());
  m.def(
      "cell_facet_normals",
      [](cell::type cell_type) {
        xt::xtensor<double, 2> normals = cell::facet_normals(cell_type);
        return py::array_t<double>(normals.shape(), normals.data());
      },
      basix::docstring::cell_facet_normals.c_str());
  m.def(
      "cell_facet_reference_volumes",
      [](cell::type cell_type) {
        xt::xtensor<double, 1> volumes
            = cell::facet_reference_volumes(cell_type);
        return py::array_t<double>(volumes.shape(), volumes.data());
      },
      basix::docstring::cell_facet_reference_volumes.c_str());
  m.def(
      "cell_facet_outward_normals",
      [](cell::type cell_type) {
        xt::xtensor<double, 2> normals = cell::facet_outward_normals(cell_type);
        return py::array_t<double>(normals.shape(), normals.data());
      },
      basix::docstring::cell_facet_outward_normals.c_str());
  m.def("cell_facet_orientations", &cell::facet_orientations,
        basix::docstring::cell_facet_orientations.c_str());
  m.def(
      "cell_facet_jacobians",
      [](cell::type cell_type) {
        xt::xtensor<double, 3> jacobians = cell::facet_jacobians(cell_type);
        return py::array_t<double>(jacobians.shape(), jacobians.data());
      },
      basix::docstring::cell_facet_jacobians.c_str());

  py::enum_<element::family>(m, "ElementFamily")
      .value("P", element::family::P)
      .value("BDM", element::family::BDM)
      .value("RT", element::family::RT)
      .value("N1E", element::family::N1E)
      .value("N2E", element::family::N2E)
      .value("Regge", element::family::Regge)
      .value("bubble", element::family::bubble)
      .value("serendipity", element::family::serendipity)
      .value("DPC", element::family::DPC)
      .value("CR", element::family::CR);

  py::class_<FiniteElement>(m, "FiniteElement", "Finite Element")
      .def(
          "tabulate",
          [](const FiniteElement& self, int n,
             const py::array_t<double, py::array::c_style>& x) {
            auto _x = adapt_x(x);
            auto t = self.tabulate(n, _x);
            return py::array_t<double>(t.shape(), t.data());
          },
          basix::docstring::FiniteElement__tabulate.c_str())
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
          basix::docstring::FiniteElement__map_push_forward.c_str())
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
          basix::docstring::FiniteElement__map_pull_back.c_str())
      .def(
          "apply_dof_transformation",
          [](const FiniteElement& self, py::array_t<double>& data,
             int block_size, std::uint32_t cell_info) {
            xtl::span<double> data_span(data.mutable_data(), data.size());
            self.apply_dof_transformation(data_span, block_size, cell_info);
            return py::array_t<double>(data_span.size(), data_span.data());
          },
          basix::docstring::FiniteElement__apply_dof_transformation.c_str())
      .def(
          "apply_dof_transformation_to_transpose",
          [](const FiniteElement& self, py::array_t<double>& data,
             int block_size, std::uint32_t cell_info) {
            xtl::span<double> data_span(data.mutable_data(), data.size());
            self.apply_dof_transformation_to_transpose(data_span, block_size,
                                                       cell_info);
            return py::array_t<double>(data_span.size(), data_span.data());
          },
          basix::docstring::FiniteElement__apply_dof_transformation_to_transpose
              .c_str())
      .def(
          "apply_inverse_transpose_dof_transformation",
          [](const FiniteElement& self, py::array_t<double>& data,
             int block_size, std::uint32_t cell_info) {
            xtl::span<double> data_span(data.mutable_data(), data.size());
            self.apply_inverse_transpose_dof_transformation(
                data_span, block_size, cell_info);
            return py::array_t<double>(data_span.size(), data_span.data());
          },
          basix::docstring::
              FiniteElement__apply_inverse_transpose_dof_transformation.c_str())
      .def(
          "base_transformations",
          [](const FiniteElement& self) {
            xt::xtensor<double, 3> t = self.base_transformations();
            return py::array_t<double>(t.shape(), t.data());
          },
          basix::docstring::FiniteElement__base_transformations.c_str())
      .def(
          "entity_transformations",
          [](const FiniteElement& self) {
            std::map<cell::type, xt::xtensor<double, 3>> t
                = self.entity_transformations();
            py::dict t2;
            for (auto tpart : t)
            {
              t2[cell_type_to_str(tpart.first).c_str()] = py::array_t<double>(
                  tpart.second.shape(), tpart.second.data());
            }
            return t2;
          },
          basix::docstring::FiniteElement__entity_transformations.c_str())
      .def_property_readonly("degree", &FiniteElement::degree)
      .def_property_readonly("cell_type", &FiniteElement::cell_type)
      .def_property_readonly("dim", &FiniteElement::dim)
      .def_property_readonly("num_entity_dofs", &FiniteElement::num_entity_dofs)
      .def_property_readonly("entity_dofs", &FiniteElement::entity_dofs)
      .def_property_readonly("num_entity_closure_dofs",
                             &FiniteElement::num_entity_closure_dofs)
      .def_property_readonly("entity_closure_dofs",
                             &FiniteElement::entity_closure_dofs)
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
          "interpolation_matrix",
          [](const FiniteElement& self) {
            const xt::xtensor<double, 2>& P = self.interpolation_matrix();
            return py::array_t<double>(P.shape(), P.data(), py::cast(self));
          })
      .def_property_readonly(
          "dual_matrix",
          [](const FiniteElement& self) {
            const xt::xtensor<double, 2>& P = self.dual_matrix();
            return py::array_t<double>(P.shape(), P.data(), py::cast(self));
          })
      .def_property_readonly(
          "coefficient_matrix", [](const FiniteElement& self) {
            const xt::xtensor<double, 2>& P = self.coefficient_matrix();
            return py::array_t<double>(P.shape(), P.data(), py::cast(self));
          });

  py::enum_<element::lagrange_variant>(m, "LagrangeVariant")
      .value("equispaced", element::lagrange_variant::equispaced)
      .value("gll_warped", element::lagrange_variant::gll_warped)
      .value("gll_isaac", element::lagrange_variant::gll_isaac)
      .value("gll_centroid", element::lagrange_variant::gll_centroid)
      .value("chebyshev_warped", element::lagrange_variant::chebyshev_warped)
      .value("chebyshev_isaac", element::lagrange_variant::chebyshev_isaac)
      .value("chebyshev_centroid",
             element::lagrange_variant::chebyshev_centroid)
      .value("gl_warped", element::lagrange_variant::gl_warped)
      .value("gl_isaac", element::lagrange_variant::gl_isaac)
      .value("gl_centroid", element::lagrange_variant::gl_centroid)
      .value("vtk", element::lagrange_variant::vtk);

  // Create FiniteElement
  m.def(
      "create_element",
      [](element::family family_name, cell::type cell_name, int degree,
         bool discontinuous) -> FiniteElement {
        return basix::create_element(family_name, cell_name, degree,
                                     discontinuous);
      },
      basix::docstring::create_element__family_cell_degree_discontinuous
          .c_str());

  m.def(
      "create_element",
      [](element::family family_name, cell::type cell_name, int degree,
         element::lagrange_variant variant,
         bool discontinuous) -> FiniteElement {
        return basix::create_element(family_name, cell_name, degree, variant,
                                     discontinuous);
      },
      basix::docstring::create_element__family_cell_degree_variant_discontinuous
          .c_str());

  m.def(
      "create_element",
      [](element::family family_name, cell::type cell_name,
         int degree) -> FiniteElement {
        return basix::create_element(family_name, cell_name, degree);
      },
      basix::docstring::create_element__family_cell_degree.c_str());

  m.def(
      "create_element",
      [](element::family family_name, cell::type cell_name, int degree,
         element::lagrange_variant variant) -> FiniteElement {
        return basix::create_element(family_name, cell_name, degree, variant);
      },
      basix::docstring::create_element__family_cell_degree_variant.c_str());

  // Interpolate between elements
  m.def(
      "compute_interpolation_operator",
      [](const FiniteElement& element_from, const FiniteElement& element_to)
          -> const py::array_t<double, py::array::c_style> {
        xt::xtensor<double, 2> out
            = basix::compute_interpolation_operator(element_from, element_to);
        return py::array_t<double>(out.shape(), out.data());
      },
      basix::docstring::compute_interpolation_operator.c_str());

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
      basix::docstring::tabulate_polynomial_set.c_str());

  m.def(
      "make_quadrature",
      [](quadrature::type rule, cell::type celltype, int m) {
        auto [pts, w] = quadrature::make_quadrature(rule, celltype, m);
        // FIXME: it would be more elegant to handle 1D case as a 1D
        // array, but FFCx would need updating
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
      basix::docstring::make_quadrature__rule_celltype_m.c_str());

  m.def(
      "make_quadrature",
      [](cell::type celltype, int m) {
        auto [pts, w] = quadrature::make_quadrature(celltype, m);
        // FIXME: it would be more elegant to handle 1D case as a 1D
        // array, but FFCx would need updating
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
      basix::docstring::make_quadrature__celltype_m.c_str());

  m.def("index", py::overload_cast<int>(&basix::indexing::idx),
        basix::docstring::index__p.c_str())
      .def("index", py::overload_cast<int, int>(&basix::indexing::idx),
           basix::docstring::index__p_q.c_str())
      .def("index", py::overload_cast<int, int, int>(&basix::indexing::idx),
           basix::docstring::index__p_q_r.c_str());
}
