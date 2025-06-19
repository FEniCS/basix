// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "polynomials.h"
#include "indexing.h"
#include "mdspan.hpp"
#include "polyset.h"
#include <concepts>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace basix;
using namespace basix::indexing;

namespace
{
template <typename T, std::size_t d>
using mdarray_t = mdex::mdarray<T, md::dextents<std::size_t, d>>;
template <typename T, std::size_t d>
using mdspan_t = md::mdspan<T, md::dextents<std::size_t, d>>;

//-----------------------------------------------------------------------------
constexpr int single_choose(int n, int k)
{
  int out = 1;
  for (int i = n + 1 - k; i <= n; ++i)
    out *= i;
  for (int i = 1; i <= k; ++i)
    out /= i;
  return out;
}
//-----------------------------------------------------------------------------
int choose(int n, const std::vector<int>& powers)
{
  int out = 1;
  for (std::size_t i = 0; i < powers.size(); ++i)
  {
    out *= single_choose(n, powers[i]);
    n -= powers[i];
  }
  return out;
}
//-----------------------------------------------------------------------------
/// Compute coefficients in the Jacobi Polynomial recurrence relation
template <typename T>
constexpr std::array<T, 3> jrc(int a, int n)
{
  T an = (a + 2 * n + 1) * (a + 2 * n + 2)
         / static_cast<T>(2 * (n + 1) * (a + n + 1));
  T bn = a * a * (a + 2 * n + 1)
         / static_cast<T>(2 * (n + 1) * (a + n + 1) * (a + 2 * n));
  T cn = n * (a + n) * (a + 2 * n + 2)
         / static_cast<T>((n + 1) * (a + n + 1) * (a + 2 * n));
  return {an, bn, cn};
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
tabulate_bernstein(cell::type celltype, int d, mdspan_t<const T, 2> x)
{
  if (celltype != cell::type::interval and celltype != cell::type::triangle
      and celltype != cell::type::tetrahedron)
  {
    throw std::runtime_error("not implemented yet");
  }

  // TODO: implement a better Bernstein evaluation algorithm here

  const std::size_t pdim = dim(polynomials::type::bernstein, celltype, d);

  std::array<std::size_t, 2> shape = {pdim, x.extent(0)};
  std::vector<T> values_b(shape[0] * shape[1]);
  mdspan_t<T, 2> values(values_b.data(), shape);

  mdarray_t<T, 2> lambdas(x.extent(1) + 1, x.extent(0));
  for (std::size_t j = 0; j < lambdas.extent(1); ++j)
    lambdas(0, j) = 1.0;
  for (std::size_t i = 0; i < x.extent(1); ++i)
  {
    for (std::size_t j = 0; j < x.extent(0); ++j)
    {
      lambdas(0, j) -= x(j, i);
      lambdas(i + 1, j) = x(j, i);
    }
  }

  std::vector<int> powers(lambdas.extent(0), 0);
  powers[0] = d;

  int n = 0;
  while (powers[0] >= 0)
  {
    {
      const int p = choose(d, powers);
      for (std::size_t j = 0; j < values.extent(1); ++j)
        values(n, j) = p;
    }

    for (std::size_t l = 0; l < lambdas.extent(0); ++l)
      for (int a = 0; a < powers[l]; ++a)
        for (std::size_t j = 0; j < values.extent(1); ++j)
          values(n, j) *= lambdas(l, j);

    powers[0] -= 1;
    powers[1] += 1;
    for (std::size_t i = 1; powers[0] < 0 and i + 1 < powers.size(); ++i)
    {
      powers[i] = 0;
      powers[i + 1] += 1;
      powers[0] = d;
      for (std::size_t j = 1; j < powers.size(); ++j)
        powers[0] -= powers[j];
    }

    ++n;
  }

  return {std::move(values_b), shape};
}
//-----------------------------------------------------------------------------
template <typename T>
void tabulate_lagrange_pyramid(
    md::mdspan<T, md::dextents<std::size_t, 2>> P, std::size_t n,
    md::mdspan<const T, md::dextents<std::size_t, 2>> x)
{
  // The recurrence formulae used in this function are derived in
  // https://doi.org/10.5281/zenodo.15281516 (Scroggs, 2025)
  assert(x.extent(1) == 3);
  assert(P.extent(0) == (n + 1) * (n + 2) * (2 * n + 3) / 6);
  assert(P.extent(1) == x.extent(0));

  // Indexing for pyramidal basis functions
  auto pyr_idx = [n](std::size_t p, std::size_t q, std::size_t r) -> std::size_t
  {
    std::size_t rv = n - r + 1;
    std::size_t r0 = r * (n + 1) * (n - r + 2) + (2 * r - 1) * (r - 1) * r / 6;
    return r0 + p * rv + q;
  };

  const auto x0 = md::submdspan(x, md::full_extent, 0);
  const auto x1 = md::submdspan(x, md::full_extent, 1);
  const auto x2 = md::submdspan(x, md::full_extent, 2);

  // Traverse derivatives in increasing order
  std::fill(P.data_handle(), P.data_handle() + P.size(), 0.0);

  if (n == 0)
  {
    for (std::size_t j = 0; j < P.extent(1); ++j)
      P(pyr_idx(0, 0, 0), j) = std::sqrt(3);
    return;
  }

  for (std::size_t j = 0; j < P.extent(1); ++j)
    P(pyr_idx(0, 0, 0), j) = 1.0;

  // r = 0
  for (std::size_t p = 0; p <= n; ++p)
  {
    if (p > 0)
    {
      const T a = static_cast<T>(p - 1) / static_cast<T>(p);
      auto p00 = md::submdspan(P, pyr_idx(p, 0, 0), md::full_extent);
      auto p1 = md::submdspan(P, pyr_idx(p - 1, 0, 0), md::full_extent);
      for (std::size_t i = 0; i < p00.size(); ++i)
        p00[i] = (a + 1.0) * (x0[i] * 2.0 + x2[i] - 1.0) * p1[i];

      if (p > 1)
      {
        auto p2 = md::submdspan(P, pyr_idx(p - 2, 0, 0), md::full_extent);
        for (std::size_t i = 0; i < p00.size(); ++i)
        {
          T f2 = 1.0 - x2[i];
          p00[i] -= a * f2 * f2 * p2[i];
        }
      }
    }

    for (std::size_t q = 1; q < n + 1; ++q)
    {
      const T a = static_cast<T>(q - 1) / static_cast<T>(q);
      auto r_pq = md::submdspan(P, pyr_idx(p, q, 0), md::full_extent);

      auto _p = md::submdspan(P, pyr_idx(p, q - 1, 0), md::full_extent);
      if (q <= p)
      {
        for (std::size_t i = 0; i < r_pq.size(); ++i)
        {
          const T x1over = x2[i] == 1.0 ? 0.0 : x1[i] / (1.0 - x2[i]);
          r_pq[i] = (a + 1.0) * (2.0 * x1over - 1.0) * _p[i];
        }
      }
      else
      {
        for (std::size_t i = 0; i < r_pq.size(); ++i)
          r_pq[i] = (a + 1.0) * (2.0 * x1[i] + x2[i] - 1.0) * _p[i];
      }

      if (q > 1)
      {
        auto _p = md::submdspan(P, pyr_idx(p, q - 2, 0), md::full_extent);
        if (q <= p)
        {
          for (std::size_t i = 0; i < r_pq.size(); ++i)
            r_pq[i] -= a * _p[i];
        }
        else if (q == p + 1)
        {
          for (std::size_t i = 0; i < r_pq.size(); ++i)
            r_pq[i] -= a * (1.0 - x2[i]) * _p[i];
        }
        else
        {
          for (std::size_t i = 0; i < r_pq.size(); ++i)
          {
            const T f2 = 1.0 - x2[i];
            r_pq[i] -= a * f2 * f2 * _p[i];
          }
        }
      }
    }
        }

        // Extend into r > 0
        for (std::size_t r = 1; r <= n; ++r)
        {
          for (std::size_t p = 0; p <= n - r; ++p)
          {
            for (std::size_t q = 0; q <= n - r; ++q)
            {
              auto [ar, br, cr] = jrc<T>(2 * std::max(p, q) + 2, r - 1);
              auto r_pqr = md::submdspan(P, pyr_idx(p, q, r), md::full_extent);
              auto _r0
                  = md::submdspan(P, pyr_idx(p, q, r - 1), md::full_extent);
              for (std::size_t i = 0; i < r_pqr.size(); ++i)
                r_pqr[i] = _r0[i] * (2.0 * x2[i] * ar + br - ar);
              if (r > 1)
              {
                auto _r
                    = md::submdspan(P, pyr_idx(p, q, r - 2), md::full_extent);
                for (std::size_t i = 0; i < r_pqr.size(); ++i)
                  r_pqr[i] -= _r[i] * cr;
              }
            }
          }
        }

  for (std::size_t r = 0; r <= n; ++r)
  {
    for (std::size_t p = 0; p <= n - r; ++p)
    {
      for (std::size_t q = 0; q <= n - r; ++q)
      {
        auto pqr = md::submdspan(P, pyr_idx(p, q, r), md::full_extent);
        for (std::size_t i = 0; i < pqr.extent(0); ++i)
          pqr(i) *= std::sqrt(2 * (q + 0.5) * (p + 0.5)
                              * (std::max(p, q) + r + 1.5))
                    * 2;
      }
    }
  }
}
//-----------------------------------------------------------------------------
} // namespace

//-----------------------------------------------------------------------------
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
polynomials::tabulate(polynomials::type polytype, cell::type celltype, int d,
                      md::mdspan<const T, md::dextents<std::size_t, 2>> x)
{
  switch (polytype)
  {
  case polynomials::type::legendre:
  {
    auto [values, shape]
        = polyset::tabulate(celltype, polyset::type::standard, d, 0, x);
    assert(shape[0] == 1);
    return {std::move(values), {shape[1], shape[2]}};
  }
  case polynomials::type::lagrange:
  {
    if (celltype == cell::type::pyramid)
    {
      std::array<std::size_t, 2> shape
          = {(std::size_t)dim(polynomials::type::lagrange, celltype, d),
             x.extent(0)};
      std::vector<T> P(shape[0] * shape[1]);
      md::mdspan<T, md::dextents<std::size_t, 2>> _P(P.data(), shape);

      tabulate_lagrange_pyramid(_P, d, x);
      return {std::move(P), {shape[0], shape[1]}};
    }
    else
    {
      auto [values, shape]
          = polyset::tabulate(celltype, polyset::type::standard, d, 0, x);
      assert(shape[0] == 1);
      return {std::move(values), {shape[1], shape[2]}};
    }
  }
  case polynomials::type::bernstein:
  {
    auto [values, shape] = tabulate_bernstein(celltype, d, x);
    return {std::move(values), std::move(shape)};
  }
  default:
    throw std::runtime_error("not implemented yet");
  }
}
//-----------------------------------------------------------------------------
int polynomials::dim(polynomials::type ptype, cell::type celltype, int d)
{
  if (ptype == polynomials::type::lagrange && celltype == cell::type::pyramid)
    return (d + 1) * (d + 2) * (2 * d + 3) / 6;
  return polyset::dim(celltype, polyset::type::standard, d);
}
//-----------------------------------------------------------------------------
/// @cond
template std::pair<std::vector<float>, std::array<std::size_t, 2>>
polynomials::tabulate(polynomials::type, cell::type, int,
                      md::mdspan<const float, md::dextents<std::size_t, 2>>);
template std::pair<std::vector<double>, std::array<std::size_t, 2>>
polynomials::tabulate(polynomials::type, cell::type, int,
                      md::mdspan<const double, md::dextents<std::size_t, 2>>);
/// @endcond
//-----------------------------------------------------------------------------
