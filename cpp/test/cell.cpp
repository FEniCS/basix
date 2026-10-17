// Copyright (c) 2026 Garth N. Wells
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include <basix/cell.h>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <cstddef>
#include <vector>

using namespace basix;

TEST_CASE("Cell topology is consistent with the sub-entity counts", "[cell]")
{
  auto celltype = GENERATE(cell::type::point, cell::type::interval,
                           cell::type::triangle, cell::type::tetrahedron,
                           cell::type::quadrilateral, cell::type::hexahedron,
                           cell::type::prism, cell::type::pyramid);
  const int tdim = cell::topological_dimension(celltype);
  auto topology = cell::topology(celltype);
  REQUIRE(topology.size() == static_cast<std::size_t>(tdim) + 1);

  for (int dim = 0; dim <= tdim; ++dim)
  {
    CHECK(static_cast<int>(topology[dim].size())
          == cell::num_sub_entities(celltype, dim));
  }

  // The cell is its own only sub-entity of dimension tdim
  CHECK(cell::num_sub_entities(celltype, tdim) == 1);
  CHECK(cell::sub_entity_type(celltype, tdim, 0) == celltype);

  // Each vertex is a sub-entity holding only itself
  for (std::size_t v = 0; v < topology[0].size(); ++v)
  {
    REQUIRE(topology[0][v].size() == 1);
    CHECK(topology[0][v][0] == static_cast<int>(v));
    CHECK(cell::sub_entity_type(celltype, 0, static_cast<int>(v))
          == cell::type::point);
  }

  // The sub-entity types returned in bulk agree with the per-entity ones
  auto types = cell::subentity_types(celltype);
  REQUIRE(types.size() == static_cast<std::size_t>(tdim) + 1);
  for (int dim = 0; dim <= tdim; ++dim)
  {
    REQUIRE(types[dim].size() == topology[dim].size());
    for (std::size_t e = 0; e < types[dim].size(); ++e)
    {
      CHECK(types[dim][e]
            == cell::sub_entity_type(celltype, dim, static_cast<int>(e)));
    }
  }
}

TEST_CASE("Cell geometry has one point per vertex", "[cell]")
{
  auto celltype = GENERATE(cell::type::interval, cell::type::triangle,
                           cell::type::tetrahedron, cell::type::quadrilateral,
                           cell::type::hexahedron, cell::type::prism,
                           cell::type::pyramid);
  const std::size_t tdim = cell::topological_dimension(celltype);
  auto [x, shape] = cell::geometry<double>(celltype);
  CHECK(shape[0]
        == static_cast<std::size_t>(cell::num_sub_entities(celltype, 0)));
  CHECK(shape[1] == tdim);
  REQUIRE(x.size() == shape[0] * shape[1]);

  // Reference cells live in the unit (hyper)cube
  for (double xi : x)
  {
    CHECK(xi >= 0.0);
    CHECK(xi <= 1.0);
  }

  // Sub-entity geometry is a subset of the cell geometry
  for (std::size_t dim = 0; dim <= tdim; ++dim)
  {
    auto topology = cell::topology(celltype);
    for (std::size_t e = 0; e < topology[dim].size(); ++e)
    {
      auto [xe, eshape] = cell::sub_entity_geometry<double>(
          celltype, static_cast<int>(dim), static_cast<int>(e));
      REQUIRE(eshape[0] == topology[dim][e].size());
      REQUIRE(eshape[1] == tdim);
      for (std::size_t i = 0; i < eshape[0]; ++i)
      {
        const int v = topology[dim][e][i];
        for (std::size_t j = 0; j < tdim; ++j)
          CHECK(xe[i * eshape[1] + j] == x[v * shape[1] + j]);
      }
    }
  }
}

TEST_CASE("Reference cell volumes", "[cell]")
{
  CHECK(cell::volume<double>(cell::type::interval) == Catch::Approx(1.0));
  CHECK(cell::volume<double>(cell::type::triangle) == Catch::Approx(0.5));
  CHECK(cell::volume<double>(cell::type::tetrahedron)
        == Catch::Approx(1.0 / 6.0));
  CHECK(cell::volume<double>(cell::type::quadrilateral) == Catch::Approx(1.0));
  CHECK(cell::volume<double>(cell::type::hexahedron) == Catch::Approx(1.0));
  CHECK(cell::volume<double>(cell::type::prism) == Catch::Approx(0.5));
  CHECK(cell::volume<double>(cell::type::pyramid) == Catch::Approx(1.0 / 3.0));
}

TEST_CASE("Facet outward normals are unit vectors pointing out of the cell",
          "[cell]")
{
  auto celltype = GENERATE(cell::type::interval, cell::type::triangle,
                           cell::type::tetrahedron, cell::type::quadrilateral,
                           cell::type::hexahedron, cell::type::prism,
                           cell::type::pyramid);
  const std::size_t tdim = cell::topological_dimension(celltype);
  auto [x, xshape] = cell::geometry<double>(celltype);
  auto facets = cell::topology(celltype)[tdim - 1];
  auto [n, nshape] = cell::facet_outward_normals<double>(celltype);
  REQUIRE(nshape[0] == facets.size());
  REQUIRE(nshape[1] == tdim);

  // Cell midpoint. The reference cells are convex, so a facet normal is
  // outward pointing iff it has a positive component along the vector
  // from the cell midpoint to the facet midpoint.
  std::vector<double> midpoint(tdim, 0.0);
  for (std::size_t v = 0; v < xshape[0]; ++v)
    for (std::size_t j = 0; j < tdim; ++j)
      midpoint[j] += x[v * xshape[1] + j] / static_cast<double>(xshape[0]);

  for (std::size_t f = 0; f < facets.size(); ++f)
  {
    std::vector<double> fmidpoint(tdim, 0.0);
    for (int v : facets[f])
      for (std::size_t j = 0; j < tdim; ++j)
      {
        fmidpoint[j]
            += x[v * xshape[1] + j] / static_cast<double>(facets[f].size());
      }

    double norm2 = 0.0, dot = 0.0;
    for (std::size_t j = 0; j < tdim; ++j)
    {
      norm2 += n[f * nshape[1] + j] * n[f * nshape[1] + j];
      dot += n[f * nshape[1] + j] * (fmidpoint[j] - midpoint[j]);
    }
    CHECK(std::sqrt(norm2) == Catch::Approx(1.0));
    CHECK(dot > 0.0);
  }

  // facet_orientations records where facet_normals disagrees with the
  // outward normals
  auto [m, mshape] = cell::facet_normals<double>(celltype);
  auto orientations = cell::facet_orientations(celltype);
  REQUIRE(orientations.size() == facets.size());
  for (std::size_t f = 0; f < facets.size(); ++f)
  {
    const double sign = orientations[f] ? -1.0 : 1.0;
    for (std::size_t j = 0; j < tdim; ++j)
    {
      CHECK(sign * m[f * mshape[1] + j]
            == Catch::Approx(n[f * nshape[1] + j]).margin(1e-14));
    }
  }
}

TEST_CASE("Cell data is the same in single and double precision", "[cell]")
{
  auto celltype = GENERATE(cell::type::interval, cell::type::triangle,
                           cell::type::tetrahedron, cell::type::quadrilateral,
                           cell::type::hexahedron, cell::type::prism,
                           cell::type::pyramid);
  CHECK(cell::volume<float>(celltype)
        == Catch::Approx(cell::volume<double>(celltype)).epsilon(1e-6));

  auto [x32, shape32] = cell::geometry<float>(celltype);
  auto [x64, shape64] = cell::geometry<double>(celltype);
  REQUIRE(shape32 == shape64);
  for (std::size_t i = 0; i < x32.size(); ++i)
    CHECK(x32[i] == Catch::Approx(x64[i]).epsilon(1e-6));
}
