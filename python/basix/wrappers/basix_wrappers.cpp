// Copyright (c) 2020-2024 Chris Richardson, Matthew Scroggs and Garth N. Wells
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "basix_wrappers/basix_wrappers.h"
#include <basix/cell.h>
#include <basix/element-families.h>
#include <basix/finite-element.h>
#include <basix/indexing.h>
#include <basix/lattice.h>
#include <basix/maps.h>
#include <basix/polynomials.h>
#include <basix/polyset.h>
#include <basix/quadrature.h>
#include <basix/sobolev-spaces.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

using namespace basix;
using namespace basix_wrappers;

NB_MODULE(_basixcpp, m)
{
  m.doc() = "Interface to the Basix C++ library.";
  m.attr("__version__") = basix::version();

  m.def("topology", &cell::topology, "celltype"_a);
  m.def(
      "geometry",
      [](cell::type celltype)
      { return as_nbarrayp(cell::geometry<double>(celltype)); },
      "celltype"_a);
  m.def(
      "sub_entity_type",
      [](cell::type celltype, int dim, int index)
      { return cell::sub_entity_type(celltype, dim, index); },
      "celltype"_a, "dim"_a, "index"_a);
  m.def("sub_entity_connectivity", &cell::sub_entity_connectivity, "celltype"_a);
  m.def(
      "sub_entity_geometry",
      [](cell::type celltype, int dim, int index)
      {
        return as_nbarrayp(
            cell::sub_entity_geometry<double>(celltype, dim, index));
      },
      "celltype"_a, "dim"_a, "index"_a);
  m.def("subentity_types", &cell::subentity_types, "celltype"_a);
  m.def("sobolev_space_intersection", &sobolev::space_intersection, "space1"_a, "space2"_a);

  nb::enum_<lattice::type>(m, "LatticeType", nb::is_arithmetic(),
                           "Lattice type.")
      .value("equispaced", lattice::type::equispaced)
      .value("gll", lattice::type::gll)
      .value("chebyshev", lattice::type::chebyshev)
      .value("gl", lattice::type::gl);
  nb::enum_<lattice::simplex_method>(
      m, "LatticeSimplexMethod", nb::is_arithmetic(), "Lattice simplex method.")
      .value("none", lattice::simplex_method::none)
      .value("warp", lattice::simplex_method::warp)
      .value("isaac", lattice::simplex_method::isaac)
      .value("centroid", lattice::simplex_method::centroid);

  nb::enum_<polynomials::type>(m, "PolynomialType", nb::is_arithmetic(),
                               "Polynomial type.")
      .value("legendre", polynomials::type::legendre)
      .value("lagrange", polynomials::type::lagrange)
      .value("bernstein", polynomials::type::bernstein);

  m.def(
      "tabulate_polynomials",
      [](polynomials::type polytype, cell::type celltype, int d,
         const nb::ndarray<const double, nb::ndim<2>, nb::c_contig>& x)
      {
        mdspan_t<const double, 2> _x(x.data(), x.shape(0), x.shape(1));
        return as_nbarrayp(polynomials::tabulate(polytype, celltype, d, _x));
      },
      "ptype"_a, "celltype"_a, "degree"_a, "pts"_a);
  m.def("polynomials_dim", &polynomials::dim, "ptype"_a, "celltype"_a, "degree"_a);
  m.def(
      "create_lattice",
      [](cell::type celltype, int n, lattice::type type, bool exterior,
         lattice::simplex_method method)
      {
        return as_nbarrayp(
            lattice::create<double>(celltype, n, type, exterior, method));
      },
      "celltype"_a, "n"_a, "ltype"_a, "exterior"_a, "method"_a);

  nb::enum_<maps::type>(m, "MapType", nb::is_arithmetic(), "Element map type.")
      .value("identity", maps::type::identity)
      .value("L2Piola", maps::type::L2Piola)
      .value("covariantPiola", maps::type::covariantPiola)
      .value("contravariantPiola", maps::type::contravariantPiola)
      .value("doubleCovariantPiola", maps::type::doubleCovariantPiola)
      .value("doubleContravariantPiola", maps::type::doubleContravariantPiola);

  nb::enum_<sobolev::space>(m, "SobolevSpace", nb::is_arithmetic(),
                            "Sobolev space.")
      .value("L2", sobolev::space::L2)
      .value("H1", sobolev::space::H1)
      .value("H2", sobolev::space::H2)
      .value("H3", sobolev::space::H3)
      .value("HInf", sobolev::space::HInf)
      .value("HDiv", sobolev::space::HDiv)
      .value("HCurl", sobolev::space::HCurl)
      .value("HEin", sobolev::space::HEin)
      .value("HDivDiv", sobolev::space::HDivDiv);

  nb::enum_<quadrature::type>(m, "QuadratureType", nb::is_arithmetic(),
                              "Quadrature type.")
      .value("default", quadrature::type::Default)
      .value("gauss_jacobi", quadrature::type::gauss_jacobi)
      .value("gll", quadrature::type::gll)
      .value("xiao_gimbutas", quadrature::type::xiao_gimbutas);

  nb::enum_<cell::type>(m, "CellType", nb::is_arithmetic(), "Cell type.")
      .value("point", cell::type::point)
      .value("interval", cell::type::interval)
      .value("triangle", cell::type::triangle)
      .value("tetrahedron", cell::type::tetrahedron)
      .value("quadrilateral", cell::type::quadrilateral)
      .value("hexahedron", cell::type::hexahedron)
      .value("prism", cell::type::prism)
      .value("pyramid", cell::type::pyramid);

  m.def("cell_volume", &cell::volume<double>, "celltype"_a);
  m.def(
      "cell_facet_normals",
      [](cell::type cell_type)
      { return as_nbarrayp(cell::facet_normals<double>(cell_type)); },
      "celltype"_a);
  m.def(
      "cell_facet_reference_volumes",
      [](cell::type cell_type)
      { return as_nbarray(cell::facet_reference_volumes<double>(cell_type)); },
      "celltype"_a);
  m.def(
      "cell_facet_outward_normals",
      [](cell::type cell_type)
      { return as_nbarrayp(cell::facet_outward_normals<double>(cell_type)); },
      "celltype"_a);
  m.def(
      "cell_facet_orientations",
      [](cell::type cell_type)
      {
        std::vector<bool> c = cell::facet_orientations(cell_type);
        std::vector<std::uint8_t> c8(c.begin(), c.end());
        return c8;
      },
      "celltype"_a);
  m.def(
      "cell_facet_jacobians",
      [](cell::type cell_type)
      { return as_nbarrayp(cell::facet_jacobians<double>(cell_type)); },
      "celltype"_a);

  m.def(
      "cell_edge_jacobians",
      [](cell::type cell_type)
      { return as_nbarrayp(cell::edge_jacobians<double>(cell_type)); },
      "celltype"_a);

  nb::enum_<element::family>(m, "ElementFamily", nb::is_arithmetic(),
                             "Finite element family.")
      .value("custom", element::family::custom)
      .value("P", element::family::P)
      .value("BDM", element::family::BDM)
      .value("RT", element::family::RT)
      .value("N1E", element::family::N1E)
      .value("N2E", element::family::N2E)
      .value("Regge", element::family::Regge)
      .value("HHJ", element::family::HHJ)
      .value("bubble", element::family::bubble)
      .value("serendipity", element::family::serendipity)
      .value("DPC", element::family::DPC)
      .value("CR", element::family::CR)
      .value("Hermite", element::family::Hermite)
      .value("iso", element::family::iso);

  nb::enum_<element::lagrange_variant>(
      m, "LagrangeVariant", nb::is_arithmetic(), "Lagrange element variant.")
      .value("unset", element::lagrange_variant::unset)
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
      .value("legendre", element::lagrange_variant::legendre)
      .value("bernstein", element::lagrange_variant::bernstein);

  nb::enum_<element::dpc_variant>(m, "DPCVariant", nb::is_arithmetic(),
                                  "DPC variant.")
      .value("unset", element::dpc_variant::unset)
      .value("simplex_equispaced", element::dpc_variant::simplex_equispaced)
      .value("simplex_gll", element::dpc_variant::simplex_gll)
      .value("horizontal_equispaced",
             element::dpc_variant::horizontal_equispaced)
      .value("horizontal_gll", element::dpc_variant::horizontal_gll)
      .value("diagonal_equispaced", element::dpc_variant::diagonal_equispaced)
      .value("diagonal_gll", element::dpc_variant::diagonal_gll)
      .value("legendre", element::dpc_variant::legendre);

  m.def(
      "create_element",
      [](element::family family_name, cell::type cell, int degree,
         element::lagrange_variant lagrange_variant,
         element::dpc_variant dpc_variant, bool discontinuous,
         const std::vector<int>& dof_ordering, char dtype)
          -> std::variant<FiniteElement<float>, FiniteElement<double>>
      {
        return dispatch_dtype<
            std::variant<FiniteElement<float>, FiniteElement<double>>>(
            dtype,
            [&]<typename T>()
            {
              return basix::create_element<T>(family_name, cell, degree,
                                               lagrange_variant, dpc_variant,
                                               discontinuous, dof_ordering);
            });
      },
      "family"_a, "celltype"_a, "degree"_a, "lagrange_variant"_a, "dpc_variant"_a,
      "discontinuous"_a, "dof_ordering"_a, "dtype"_a);

  m.def(
      "create_tp_element",
      [](element::family family_name, cell::type cell, int degree,
         element::lagrange_variant lagrange_variant,
         element::dpc_variant dpc_variant, bool discontinuous, char dtype)
          -> std::variant<FiniteElement<float>, FiniteElement<double>>
      {
        return dispatch_dtype<
            std::variant<FiniteElement<float>, FiniteElement<double>>>(
            dtype,
            [&]<typename T>()
            {
              return basix::create_tp_element<T>(family_name, cell, degree,
                                                  lagrange_variant,
                                                  dpc_variant, discontinuous);
            });
      },
      "family"_a, "celltype"_a, "degree"_a, "lagrange_variant"_a, "dpc_variant"_a,
      "discontinuous"_a, "dtype"_a);

  m.def(
      "tp_factors",
      [](element::family family_name, cell::type cell, int degree,
         element::lagrange_variant lagrange_variant,
         element::dpc_variant dpc_variant, bool discontinuous,
         const std::vector<int>& dof_ordering, char dtype)
          -> std::optional<
              std::variant<std::vector<std::vector<FiniteElement<float>>>,
                           std::vector<std::vector<FiniteElement<double>>>>>
      {
        return dispatch_dtype<std::optional<
            std::variant<std::vector<std::vector<FiniteElement<float>>>,
                         std::vector<std::vector<FiniteElement<double>>>>>>(
            dtype,
            [&]<typename T>()
            {
              return basix::tp_factors<T>(family_name, cell, degree,
                                          lagrange_variant, dpc_variant,
                                          discontinuous, dof_ordering);
            });
      },
      "family"_a, "celltype"_a, "degree"_a, "lagrange_variant"_a, "dpc_variant"_a,
      "discontinuous"_a, "dof_ordering"_a, "dtype"_a);

  m.def("tp_dof_ordering", &basix::tp_dof_ordering, "family"_a, "celltype"_a, "degree"_a,
        "lagrange_variant"_a, "dpc_variant"_a, "discontinuous"_a);
  m.def("lex_dof_ordering", &basix::lex_dof_ordering, "family"_a, "celltype"_a, "degree"_a,
        "lagrange_variant"_a, "dpc_variant"_a, "discontinuous"_a);

  nb::enum_<polyset::type>(m, "PolysetType", nb::is_arithmetic(),
                           "Polyset type.")
      .value("standard", polyset::type::standard)
      .value("macroedge", polyset::type::macroedge);

  m.def("superset", &polyset::superset, "cell"_a, "type1"_a, "type2"_a);
  m.def("restriction", &polyset::restriction, "ptype"_a, "cell"_a, "restriction_cell"_a);

  m.def(
      "make_quadrature",
      [](quadrature::type rule, cell::type celltype, polyset::type polytype,
         int m)
      {
        auto [pts, w]
            = quadrature::make_quadrature<double>(rule, celltype, polytype, m);
        std::array shape{w.size(), pts.size() / w.size()};
        return std::pair(as_nbarray(std::move(pts), shape.size(), shape.data()),
                         as_nbarray(std::move(w)));
      },
      "rule"_a, "cell"_a, "polyset_type"_a, "degree"_a);

  m.def(
      "gauss_jacobi_rule",
      [](double a, int m)
      {
        auto [pts, w] = quadrature::gauss_jacobi_rule<double>(a, m);
        return std::pair(as_nbarray(std::move(pts)),
                         as_nbarray(std::move(w)));
      },
      "alpha"_a, "npoints"_a);

  m.def("index", nb::overload_cast<int>(&basix::indexing::idx), "p"_a);
  m.def("index", nb::overload_cast<int, int>(&basix::indexing::idx), "p"_a, "q"_a);
  m.def("index", nb::overload_cast<int, int, int>(&basix::indexing::idx), "p"_a, "q"_a,
        "r"_a);

  declare_float<float>(m, "float32");
  declare_float<double>(m, "float64");
}
