// Copyright (c) 2026 Garth N. Wells
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include <algorithm>
#include <basix/cell.h>
#include <basix/element-families.h>
#include <basix/finite-element.h>
#include <basix/maps.h>
#include <basix/polyset.h>
#include <basix/quadrature.h>
#include <basix/sobolev-spaces.h>
#include <basix/types.h>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstddef>
#include <span>
#include <vector>

using namespace basix;

namespace
{
/// Create a Lagrange element of the given degree on the given cell
FiniteElement<double> lagrange(cell::type celltype, int degree)
{
  return create_element<double>(element::family::P, celltype, degree,
                                element::lagrange_variant::equispaced,
                                element::dpc_variant::unset, false);
}
} // namespace

TEST_CASE("Lagrange elements are a partition of unity", "[finite-element]")
{
  auto celltype = GENERATE(cell::type::interval, cell::type::triangle,
                           cell::type::tetrahedron, cell::type::quadrilateral,
                           cell::type::hexahedron, cell::type::prism,
                           cell::type::pyramid);
  auto degree = GENERATE(1, 2, 3);
  auto e = lagrange(celltype, degree);

  auto [pts, wts] = quadrature::make_quadrature<double>(
      quadrature::type::Default, celltype, polyset::type::standard, 3);
  const std::size_t tdim = cell::topological_dimension(celltype);

  auto [_T, shape]
      = e.tabulate(1, std::span<const double>(pts), {wts.size(), tdim});
  md::mdspan<const double, md::dextents<std::size_t, 4>> T(_T.data(), shape);
  REQUIRE(shape == e.tabulate_shape(1, wts.size()));
  REQUIRE(shape[2] == static_cast<std::size_t>(e.dim()));
  REQUIRE(shape[3] == 1);

  for (std::size_t p = 0; p < shape[1]; ++p)
  {
    // The basis functions sum to 1 ...
    double sum = 0.0;
    for (std::size_t i = 0; i < shape[2]; ++i)
      sum += T(0, p, i, 0);
    CHECK(sum == Catch::Approx(1.0).margin(1e-12));

    // ... so all of their derivatives sum to 0
    for (std::size_t d = 1; d < shape[0]; ++d)
    {
      double dsum = 0.0;
      for (std::size_t i = 0; i < shape[2]; ++i)
        dsum += T(d, p, i, 0);
      CHECK(dsum == Catch::Approx(0.0).margin(1e-10));
    }
  }
}

TEST_CASE("The Lagrange basis is a Kronecker delta at the element points",
          "[finite-element]")
{
  auto celltype = GENERATE(cell::type::interval, cell::type::triangle,
                           cell::type::tetrahedron, cell::type::quadrilateral,
                           cell::type::hexahedron, cell::type::prism,
                           cell::type::pyramid);
  auto degree = GENERATE(1, 2, 3);
  auto e = lagrange(celltype, degree);

  const auto& [x, xshape] = e.points();
  REQUIRE(xshape[0] == static_cast<std::size_t>(e.dim()));

  auto [_T, shape] = e.tabulate(0, std::span<const double>(x), xshape);
  md::mdspan<const double, md::dextents<std::size_t, 4>> T(_T.data(), shape);
  for (std::size_t p = 0; p < shape[1]; ++p)
    for (std::size_t i = 0; i < shape[2]; ++i)
      CHECK(T(0, p, i, 0) == Catch::Approx(p == i ? 1.0 : 0.0).margin(1e-11));
}

TEST_CASE("Entity DOFs partition the element degrees of freedom",
          "[finite-element]")
{
  auto celltype = GENERATE(cell::type::interval, cell::type::triangle,
                           cell::type::tetrahedron, cell::type::quadrilateral,
                           cell::type::hexahedron, cell::type::prism,
                           cell::type::pyramid);
  auto degree = GENERATE(1, 2, 3);
  auto e = lagrange(celltype, degree);

  const int tdim = cell::topological_dimension(celltype);
  const auto& edofs = e.entity_dofs();
  REQUIRE(edofs.size() == static_cast<std::size_t>(tdim) + 1);

  std::vector<int> count(e.dim(), 0);
  for (int dim = 0; dim <= tdim; ++dim)
  {
    REQUIRE(static_cast<int>(edofs[dim].size())
            == cell::num_sub_entities(celltype, dim));
    for (const auto& dofs : edofs[dim])
      for (int dof : dofs)
      {
        REQUIRE(dof >= 0);
        REQUIRE(dof < e.dim());
        ++count[dof];
      }
  }

  // Every DOF is associated with exactly one entity
  CHECK(std::ranges::all_of(count, [](int c) { return c == 1; }));

  // A degree-1 Lagrange element has exactly one DOF per vertex
  if (degree == 1)
  {
    for (const auto& dofs : edofs[0])
      CHECK(dofs.size() == 1);
  }
}

TEST_CASE("Discontinuous elements put all DOFs on the cell interior",
          "[finite-element]")
{
  auto celltype = GENERATE(cell::type::interval, cell::type::triangle,
                           cell::type::tetrahedron, cell::type::quadrilateral,
                           cell::type::hexahedron);
  auto degree = GENERATE(1, 2);
  auto e = create_element<double>(element::family::P, celltype, degree,
                                  element::lagrange_variant::equispaced,
                                  element::dpc_variant::unset, true);
  CHECK(e.discontinuous());
  CHECK(e.dim() == lagrange(celltype, degree).dim());
  CHECK(e.sobolev_space() == sobolev::space::L2);

  const int tdim = cell::topological_dimension(celltype);
  const auto& edofs = e.entity_dofs();
  for (int dim = 0; dim < tdim; ++dim)
    for (const auto& dofs : edofs[dim])
      CHECK(dofs.empty());
  REQUIRE(edofs[tdim].size() == 1);
  CHECK(static_cast<int>(edofs[tdim][0].size()) == e.dim());
}

TEST_CASE("Elements compare and hash by their definition", "[finite-element]")
{
  auto e0 = lagrange(cell::type::triangle, 2);
  auto e1 = lagrange(cell::type::triangle, 2);
  auto e2 = lagrange(cell::type::triangle, 3);
  auto e3 = lagrange(cell::type::quadrilateral, 2);

  CHECK(e0 == e1);
  CHECK(e0.hash() == e1.hash());

  CHECK(!(e0 == e2));
  CHECK(e0.hash() != e2.hash());

  CHECK(!(e0 == e3));
  CHECK(e0.hash() != e3.hash());

  // A Lagrange variant is part of the element's identity
  auto e4 = create_element<double>(element::family::P, cell::type::triangle, 2,
                                   element::lagrange_variant::gll_warped,
                                   element::dpc_variant::unset, false);
  CHECK(!(e0 == e4));
}

TEST_CASE("Vector-valued element families have the expected properties",
          "[finite-element]")
{
  auto celltype = GENERATE(cell::type::triangle, cell::type::tetrahedron);
  const std::size_t tdim = cell::topological_dimension(celltype);

  SECTION("Raviart-Thomas")
  {
    auto e = create_element<double>(element::family::RT, celltype, 1,
                                    element::lagrange_variant::unset,
                                    element::dpc_variant::unset, false);
    CHECK(e.value_shape() == std::vector<std::size_t>{tdim});
    CHECK(e.map_type() == maps::type::contravariantPiola);
    CHECK(e.sobolev_space() == sobolev::space::HDiv);
    CHECK(e.dim()
          == cell::num_sub_entities(celltype, static_cast<int>(tdim) - 1));
  }

  SECTION("Nedelec (first kind)")
  {
    auto e = create_element<double>(element::family::N1E, celltype, 1,
                                    element::lagrange_variant::unset,
                                    element::dpc_variant::unset, false);
    CHECK(e.value_shape() == std::vector<std::size_t>{tdim});
    CHECK(e.map_type() == maps::type::covariantPiola);
    CHECK(e.sobolev_space() == sobolev::space::HCurl);
    CHECK(e.dim() == cell::num_sub_entities(celltype, 1));
  }
}

TEST_CASE("Lagrange elements are H1 and embed degree-k polynomials",
          "[finite-element]")
{
  auto celltype = GENERATE(cell::type::interval, cell::type::triangle,
                           cell::type::tetrahedron, cell::type::quadrilateral,
                           cell::type::hexahedron);
  auto degree = GENERATE(1, 2, 3);
  auto e = lagrange(celltype, degree);

  CHECK(e.family() == element::family::P);
  CHECK(e.cell_type() == celltype);
  CHECK(e.degree() == degree);
  CHECK(e.sobolev_space() == sobolev::space::H1);
  CHECK(e.map_type() == maps::type::identity);
  CHECK(e.value_shape().empty());
  CHECK(!e.discontinuous());
  CHECK(e.embedded_subdegree() == degree);
  CHECK(e.embedded_superdegree() >= degree);
}
