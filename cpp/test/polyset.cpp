// Copyright (c) 2026 Garth N. Wells
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include <algorithm>
#include <basix/cell.h>
#include <basix/indexing.h>
#include <basix/polyset.h>
#include <basix/quadrature.h>
#include <basix/types.h>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cstddef>
#include <vector>

using namespace basix;

TEST_CASE("Polyset dimensions match the tabulated array", "[polyset]")
{
  auto celltype = GENERATE(cell::type::interval, cell::type::triangle,
                           cell::type::tetrahedron, cell::type::quadrilateral,
                           cell::type::hexahedron, cell::type::prism,
                           cell::type::pyramid);
  auto d = GENERATE(0, 1, 2, 3);
  auto nd = GENERATE(0, 1, 2);

  auto [pts, wts] = quadrature::make_quadrature<double>(
      quadrature::type::Default, celltype, polyset::type::standard, 2);
  const std::size_t tdim = cell::topological_dimension(celltype);
  md::mdspan<const double, md::dextents<std::size_t, 2>> x(pts.data(),
                                                           wts.size(), tdim);

  auto [P, shape]
      = polyset::tabulate<double>(celltype, polyset::type::standard, d, nd, x);
  CHECK(shape[0] == static_cast<std::size_t>(polyset::nderivs(celltype, nd)));
  CHECK(shape[1]
        == static_cast<std::size_t>(
            polyset::dim(celltype, polyset::type::standard, d)));
  CHECK(shape[2] == wts.size());
  REQUIRE(P.size() == shape[0] * shape[1] * shape[2]);
}

TEST_CASE("The orthonormal polyset is orthonormal", "[polyset]")
{
  // The polyset on a pyramid is rational, so a standard quadrature rule
  // does not integrate products of its members exactly.
  auto celltype = GENERATE(cell::type::interval, cell::type::triangle,
                           cell::type::tetrahedron, cell::type::quadrilateral,
                           cell::type::hexahedron, cell::type::prism);
  auto d = GENERATE(1, 2, 3, 4);

  auto [pts, wts] = quadrature::make_quadrature<double>(
      quadrature::type::Default, celltype, polyset::type::standard, 2 * d);
  const std::size_t tdim = cell::topological_dimension(celltype);
  md::mdspan<const double, md::dextents<std::size_t, 2>> x(pts.data(),
                                                           wts.size(), tdim);

  auto [_P, shape]
      = polyset::tabulate<double>(celltype, polyset::type::standard, d, 0, x);
  md::mdspan<const double, md::dextents<std::size_t, 3>> P(_P.data(), shape);

  for (std::size_t i = 0; i < shape[1]; ++i)
  {
    for (std::size_t j = 0; j <= i; ++j)
    {
      double integral = 0.0;
      for (std::size_t q = 0; q < shape[2]; ++q)
        integral += wts[q] * P(0, i, q) * P(0, j, q);
      CHECK(integral == Catch::Approx(i == j ? 1.0 : 0.0).margin(1e-11));
    }
  }
}

TEST_CASE("Polyset derivatives match a finite difference approximation",
          "[polyset]")
{
  auto celltype = GENERATE(cell::type::interval, cell::type::triangle,
                           cell::type::tetrahedron, cell::type::quadrilateral,
                           cell::type::hexahedron);
  const int d = 3;
  const std::size_t tdim = cell::topological_dimension(celltype);
  const double h = 1e-5;

  // A point in the interior of every reference cell considered here
  std::vector<double> x0(tdim, 0.2);

  // Build the stencil x0, x0 +/- h e_j for each direction j
  std::vector<double> pts;
  pts.insert(pts.end(), x0.begin(), x0.end());
  for (std::size_t j = 0; j < tdim; ++j)
  {
    for (double s : {h, -h})
    {
      std::vector<double> p = x0;
      p[j] += s;
      pts.insert(pts.end(), p.begin(), p.end());
    }
  }
  const std::size_t npts = 1 + 2 * tdim;
  md::mdspan<const double, md::dextents<std::size_t, 2>> x(pts.data(), npts,
                                                           tdim);

  auto [_P, shape]
      = polyset::tabulate<double>(celltype, polyset::type::standard, d, 1, x);
  md::mdspan<const double, md::dextents<std::size_t, 3>> P(_P.data(), shape);

  for (std::size_t j = 0; j < tdim; ++j)
  {
    // Index of the first derivative in direction j
    const std::size_t deriv = tdim == 1 ? indexing::idx(1)
                              : tdim == 2
                                  ? indexing::idx(j == 0, j == 1)
                                  : indexing::idx(j == 0, j == 1, j == 2);
    for (std::size_t i = 0; i < shape[1]; ++i)
    {
      const double fd = (P(0, i, 1 + 2 * j) - P(0, i, 2 + 2 * j)) / (2.0 * h);
      CHECK(P(deriv, i, 0) == Catch::Approx(fd).margin(1e-7));
    }
  }
}

TEST_CASE("Derivative indexing is triangular", "[polyset]")
{
  CHECK(indexing::idx(0) == 0);
  CHECK(indexing::idx(3) == 3);

  // 2D: (0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...
  CHECK(indexing::idx(0, 0) == 0);
  CHECK(indexing::idx(1, 0) == 1);
  CHECK(indexing::idx(0, 1) == 2);
  CHECK(indexing::idx(2, 0) == 3);
  CHECK(indexing::idx(1, 1) == 4);
  CHECK(indexing::idx(0, 2) == 5);

  // 3D: (0,0,0), (1,0,0), (0,1,0), (0,0,1), (2,0,0), ...
  CHECK(indexing::idx(0, 0, 0) == 0);
  CHECK(indexing::idx(1, 0, 0) == 1);
  CHECK(indexing::idx(0, 1, 0) == 2);
  CHECK(indexing::idx(0, 0, 1) == 3);
  CHECK(indexing::idx(2, 0, 0) == 4);

  // The index of the (n + 1)-th order derivatives starts where the
  // n-th order block ends, i.e. the map is a bijection onto [0, nderivs)
  for (int n = 0; n < 5; ++n)
  {
    std::vector<bool> seen(polyset::nderivs(cell::type::tetrahedron, n), false);
    for (int p = 0; p <= n; ++p)
    {
      for (int q = 0; p + q <= n; ++q)
      {
        for (int r = 0; p + q + r <= n; ++r)
        {
          const int i = indexing::idx(p, q, r);
          REQUIRE(i < static_cast<int>(seen.size()));
          REQUIRE(!seen[i]);
          seen[i] = true;
        }
      }
    }
    CHECK(std::ranges::all_of(seen, [](bool b) { return b; }));
  }
}
