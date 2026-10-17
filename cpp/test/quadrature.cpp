// Copyright (c) 2026 Garth N. Wells
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include <algorithm>
#include <array>
#include <basix/cell.h>
#include <basix/polyset.h>
#include <basix/quadrature.h>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace basix;

TEST_CASE("Quadrature weights sum to the reference cell volume", "[quadrature]")
{
  auto celltype = GENERATE(cell::type::interval, cell::type::triangle,
                           cell::type::tetrahedron, cell::type::quadrilateral,
                           cell::type::hexahedron, cell::type::prism,
                           cell::type::pyramid);
  auto m = GENERATE(range(1, 9));
  auto [pts, wts] = quadrature::make_quadrature<double>(
      quadrature::type::Default, celltype, polyset::type::standard, m);

  const std::size_t tdim = cell::topological_dimension(celltype);
  REQUIRE(!wts.empty());
  REQUIRE(pts.size() == wts.size() * tdim);

  // Integrating the constant 1 gives the cell volume
  CHECK(std::reduce(wts.begin(), wts.end())
        == Catch::Approx(cell::volume<double>(celltype)).margin(1e-12));

  // Points lie in the unit (hyper)cube containing the reference cell
  for (double x : pts)
  {
    CHECK(x >= -1e-12);
    CHECK(x <= 1.0 + 1e-12);
  }
}

TEST_CASE("Gauss-Jacobi rules are exact to degree 2m - 1", "[quadrature]")
{
  // gauss_jacobi_rule(a, m) integrates f(x) (1 - x)^a over [0, 1], and is
  // exact for f a polynomial of degree 2m - 1. For f = x^k the exact value
  // is the Beta function B(k + 1, a + 1) = k! a! / (k + a + 1)!.
  const int m = 5;
  auto a = GENERATE(0, 1, 2);
  auto [pts, wts]
      = quadrature::gauss_jacobi_rule<double>(static_cast<double>(a), m);
  REQUIRE(pts.size() == static_cast<std::size_t>(m));
  REQUIRE(wts.size() == static_cast<std::size_t>(m));

  for (int k = 0; k < 2 * m; ++k)
  {
    double q = 0.0;
    for (std::size_t i = 0; i < pts.size(); ++i)
      q += wts[i] * std::pow(pts[i], k);

    double exact = 1.0;
    for (int i = 1; i <= a; ++i)
      exact *= i;
    for (int i = 1; i <= a + 1; ++i)
      exact /= (k + i);

    CHECK(q == Catch::Approx(exact).margin(1e-13));
  }
}

TEST_CASE("Gauss-Lobatto-Legendre points include the interval endpoints",
          "[quadrature]")
{
  auto m = GENERATE(range(2, 7));
  auto pts = quadrature::get_gll_points<double>(m);
  REQUIRE(pts.size() == static_cast<std::size_t>(m));

  std::ranges::sort(pts);
  CHECK(pts.front() == Catch::Approx(0.0).margin(1e-14));
  CHECK(pts.back() == Catch::Approx(1.0).margin(1e-14));

  // The points are symmetric about the midpoint of the interval
  for (std::size_t i = 0; i < pts.size(); ++i)
    CHECK(pts[i] + pts[pts.size() - 1 - i] == Catch::Approx(1.0).margin(1e-14));
}

TEST_CASE("Gauss-Legendre points lie inside the interval", "[quadrature]")
{
  auto m = GENERATE(range(1, 7));
  auto pts = quadrature::get_gl_points<double>(m);
  REQUIRE(pts.size() == static_cast<std::size_t>(m));

  std::ranges::sort(pts);
  CHECK(pts.front() > 0.0);
  CHECK(pts.back() < 1.0);
  for (std::size_t i = 0; i < pts.size(); ++i)
    CHECK(pts[i] + pts[pts.size() - 1 - i] == Catch::Approx(1.0).margin(1e-14));
}

TEST_CASE("The default quadrature rule agrees with the rule it selects",
          "[quadrature]")
{
  auto celltype = GENERATE(cell::type::interval, cell::type::triangle,
                           cell::type::tetrahedron, cell::type::quadrilateral,
                           cell::type::hexahedron);
  auto m = GENERATE(range(1, 9));

  auto [p0, w0] = quadrature::make_quadrature<double>(
      quadrature::type::Default, celltype, polyset::type::standard, m);
  auto [p1, w1] = quadrature::make_quadrature<double>(
      quadrature::get_default_rule(celltype, m), celltype,
      polyset::type::standard, m);

  REQUIRE(p0.size() == p1.size());
  REQUIRE(w0.size() == w1.size());
  for (std::size_t i = 0; i < p0.size(); ++i)
    CHECK(p0[i] == Catch::Approx(p1[i]).margin(1e-14));
  for (std::size_t i = 0; i < w0.size(); ++i)
    CHECK(w0[i] == Catch::Approx(w1[i]).margin(1e-14));
}

namespace
{
/// Exact integral of x^a y^b z^c over a Basix reference cell
double monomial_integral(cell::type celltype, std::array<int, 3> e)
{
  auto factorial = [](int n)
  {
    double f = 1.0;
    for (int i = 2; i <= n; ++i)
      f *= i;
    return f;
  };

  auto [a, b, c] = e;
  switch (celltype)
  {
  case cell::type::triangle:
    return factorial(a) * factorial(b) / factorial(a + b + 2);
  case cell::type::quadrilateral:
    return 1.0 / ((a + 1) * (b + 1));
  case cell::type::tetrahedron:
    return factorial(a) * factorial(b) * factorial(c)
           / factorial(a + b + c + 3);
  case cell::type::hexahedron:
    return 1.0 / ((a + 1) * (b + 1) * (c + 1));
  case cell::type::prism:
    return factorial(a) * factorial(b) / (factorial(a + b + 2) * (c + 1));
  case cell::type::pyramid:
    return factorial(c) * factorial(a + b + 2)
           / ((a + 1) * (b + 1) * factorial(a + b + c + 3));
  default:
    throw std::runtime_error("Unsupported cell type");
  }
}
} // namespace

TEST_CASE("Witherden-Vincent rules are exact to their degree", "[quadrature]")
{
  auto [celltype, max_degree] = GENERATE(
      std::pair{cell::type::triangle, 20},
      std::pair{cell::type::quadrilateral, 21},
      std::pair{cell::type::tetrahedron, 10},
      std::pair{cell::type::hexahedron, 11}, std::pair{cell::type::prism, 10},
      std::pair{cell::type::pyramid, 10});
  auto m = GENERATE_COPY(range(1, max_degree + 1));

  auto [pts, wts] = quadrature::make_quadrature<double>(
      quadrature::type::witherden_vincent, celltype, polyset::type::standard,
      m);
  const std::size_t tdim = cell::topological_dimension(celltype);
  REQUIRE(pts.size() == wts.size() * tdim);

  // All weights are positive and sum to the reference cell volume
  CHECK(std::ranges::all_of(wts, [](double w) { return w > 0.0; }));
  CHECK(std::reduce(wts.begin(), wts.end())
        == Catch::Approx(cell::volume<double>(celltype)).margin(1e-12));

  // Monomials of total degree at most m are integrated exactly
  for (int a = 0; a <= m; ++a)
  {
    for (int b = 0; b <= m - a; ++b)
    {
      for (int c = 0; c <= (tdim == 3 ? m - a - b : 0); ++c)
      {
        double q = 0.0;
        for (std::size_t i = 0; i < wts.size(); ++i)
        {
          double f
              = std::pow(pts[i * tdim], a) * std::pow(pts[i * tdim + 1], b);
          if (tdim == 3)
            f *= std::pow(pts[i * tdim + 2], c);
          q += wts[i] * f;
        }
        CHECK(q
              == Catch::Approx(monomial_integral(celltype, {a, b, c}))
                     .margin(1e-12));
      }
    }
  }
}

TEST_CASE("Witherden-Vincent rules reject unsupported cases", "[quadrature]")
{
  CHECK_THROWS(quadrature::make_quadrature<double>(
      quadrature::type::witherden_vincent, cell::type::hexahedron,
      polyset::type::standard, 12));
  CHECK_THROWS(quadrature::make_quadrature<double>(
      quadrature::type::witherden_vincent, cell::type::point,
      polyset::type::standard, 1));
}
