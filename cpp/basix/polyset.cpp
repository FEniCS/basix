// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "polyset.h"
#include "cell.h"
#include "indexing.h"
#include "mdspan.hpp"
#include <array>
#include <cmath>
#include <stdexcept>

using namespace basix;
using namespace basix::indexing;

namespace stdex
    = MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE;

namespace
{
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
// At a point, only the constant polynomial can be used. This has value
// 1 and derivative 0.
template <typename T>
void tabulate_polyset_point_derivs(
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 3>>
        P,
    std::size_t, std::size_t nderiv,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        x)
{
  assert(x.extent(0) > 0);
  assert(P.extent(0) == nderiv + 1);
  assert(P.extent(1) == 1);
  assert(P.extent(2) == x.extent(0));

  std::fill(P.data_handle(), P.data_handle() + P.size(), 0.0);
  for (std::size_t i = 0; i < P.extent(2); ++i)
    P(0, 0, i) = 1.0;
}
//-----------------------------------------------------------------------------

/// Compute the complete set of derivatives from 0 to nderiv, for all
/// the polynomials up to order n on a line segment. The polynomials
/// used are Legendre Polynomials, with the recurrence relation given by
/// n P(n) = (2n - 1) x P_{n-1} - (n - 1) P_{n-2} in the interval [-1,
/// 1]. The range is rescaled here to [0, 1].
template <typename T>
void tabulate_polyset_line_derivs(
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 3>>
        P,
    std::size_t n, std::size_t nderiv,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        x)
{
  assert(x.extent(0) > 0);
  assert(P.extent(0) == nderiv + 1);
  assert(P.extent(1) == n + 1);
  assert(P.extent(2) == x.extent(0));

  std::fill(P.data_handle(), P.data_handle() + P.size(), 0.0);
  for (std::size_t j = 0; j < P.extent(2); ++j)
    P(0, 0, j) = 1.0;

  if (n == 0)
    return;

  auto x0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

  for (std::size_t i = 0; i < P.extent(2); ++i)
    P(0, 1, i) = (x0[i] * 2.0 - 1.0) * P(0, 0, i);

  for (std::size_t p = 2; p <= n; ++p)
  {
    const T a = 1.0 - 1.0 / static_cast<T>(p);
    for (std::size_t i = 0; i < P.extent(2); ++i)
    {
      P(0, p, i) = (x0[i] * 2.0 - 1.0) * P(0, p - 1, i) * (a + 1.0)
                   - P(0, p - 2, i) * a;
    }
  }

  for (std::size_t k = 1; k <= nderiv; ++k)
  {
    for (std::size_t p = 1; p <= n; ++p)
    {
      const T a = 1.0 - 1.0 / static_cast<T>(p);
      for (std::size_t i = 0; i < P.extent(2); ++i)
      {
        P(k, p, i) = (x0[i] * 2.0 - 1.0) * P(k, p - 1, i) * (a + 1.0)
                     + 2 * k * P(k - 1, p - 1, i) * (a + 1.0)
                     - P(k, p - 2, i) * a;
      }
    }
  }

  // Normalise
  for (std::size_t p = 0; p < P.extent(1); ++p)
  {
    const T norm = std::sqrt(2 * p + 1);
    for (std::size_t i = 0; i < P.extent(0); ++i)
      for (std::size_t j = 0; j < P.extent(2); ++j)
        P(i, p, j) *= norm;
  }
}
//-----------------------------------------------------------------------------

/// Compute the complete set of derivatives from 0 to nderiv, for all
/// the piecewise polynomials up to order n on a line segment split into two
/// parts.
template <typename T>
void tabulate_polyset_line_macroedge_derivs(
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 3>>
        P,
    std::size_t n, std::size_t nderiv,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        x)
{
  assert(x.extent(0) > 0);
  assert(P.extent(0) == nderiv + 1);
  assert(P.extent(1) == 2 * n + 1);
  assert(P.extent(2) == x.extent(0));

  auto x0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);

  std::fill(P.data_handle(), P.data_handle() + P.size(), 0.0);

  std::vector<T> factorials(n + 1, 0.0);

  for (std::size_t k = 0; k <= n; ++k)
  {
    factorials[k] = (k % 2 == 0 ? 1 : -1) * single_choose(2 * n + 1 - k, n - k)
                    * single_choose(n, k) * pow(2, n - k);
  }
  for (std::size_t d = 0; d <= nderiv; ++d)
  {
    for (std::size_t p = 0; p < P.extent(2); ++p)
    {
      T value = 0.0;
      if (x0[p] <= 0.5)
      {
        for (std::size_t k = 0; k + d <= n; ++k)
        {
          T x_term = pow(x0[p], n - k - d);
          for (std::size_t i = n - k; i > n - k - d; --i)
            x_term *= i;
          value += factorials[k] * x_term;
        }
      }
      else
      {
        for (std::size_t k = 0; k + d <= n; ++k)
        {
          T x_term = pow(1.0 - x0[p], n - k - d);
          for (std::size_t i = n - k; i > n - k - d; --i)
            x_term *= -i;
          value += factorials[k] * x_term;
        }
      }
      P(d, 0, p) = value;
    }
  }

  for (std::size_t j = 0; j < n; ++j)
  {
    for (std::size_t k = 0; k <= j; ++k)
    {
      factorials[k] = (k % 2 == 0 ? 1 : -1)
                      * single_choose(2 * n + 1 - k, j - k)
                      * single_choose(j, k) * pow(2, j - k) * pow(2, n - j)
                      * sqrt(4 * (n - j) + 2);
    }
    for (std::size_t d = 0; d <= nderiv; ++d)
    {
      for (std::size_t p = 0; p < P.extent(2); ++p)
      {
        if (x0[p] <= 0.5)
        {
          T value = 0.0;
          for (std::size_t k = 0; k + d <= j; ++k)
          {
            T x_term = pow(x0[p], j - k - d);
            for (std::size_t i = j - k; i > j - k - d; --i)
              x_term *= i;
            value += factorials[k] * x_term;
          }
          value *= pow(0.5 - x0[p], n - j - d);
          for (std::size_t i = n - j; i > n - j - d; --i)
            value *= -i;
          P(d, j + 1, p) = value;
        }
        else
        {
          T value = 0.0;
          for (std::size_t k = 0; k + d <= j; ++k)
          {
            T x_term = pow(1.0 - x0[p], j - k - d);
            for (std::size_t i = j - k; i > j - k - d; --i)
              x_term *= -i;
            value += factorials[k] * x_term;
          }
          value *= pow(x0[p] - 0.5, n - j - d);
          for (std::size_t i = n - j; i > n - j - d; --i)
            value *= i;
          P(d, j + n + 1, p) = value;
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------

/// Compute the complete set of derivatives from 0 to nderiv, for all
/// the piecewise polynomials up to order n on a quadrilateral split into 4 by
/// splitting each edge into two parts
template <typename T>
void tabulate_polyset_quadrilateral_macroedge_derivs(
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 3>>
        P,
    std::size_t n, std::size_t nderiv,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        x)
{
  assert(x.extent(0) > 0);
  assert(P.extent(0) == (nderiv + 1) * (nderiv + 2) / 2);
  assert(P.extent(1) == (2 * n + 1) * (2 * n + 1));
  assert(P.extent(2) == x.extent(0));

  // Indexing for quadrilateral basis functions
  auto quad_idx = [n](std::size_t px, std::size_t py) -> std::size_t
  { return (2 * n + 1) * px + py; };

  auto x0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
  auto x1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 1);

  std::fill(P.data_handle(), P.data_handle() + P.size(), 0.0);

  std::vector<T> factorials(n + 1, 0.0);

  // Fill with values of polynomials in x
  for (std::size_t k = 0; k <= n; ++k)
  {
    factorials[k] = (k % 2 == 0 ? 1 : -1) * single_choose(2 * n + 1 - k, n - k)
                    * single_choose(n, k) * pow(2, n - k);
  }
  for (std::size_t dx = 0; dx <= nderiv; ++dx)
  {
    for (std::size_t p = 0; p < P.extent(2); ++p)
    {
      T value = 0.0;
      if (x0[p] <= 0.5)
      {
        for (std::size_t k = 0; k + dx <= n; ++k)
        {
          T x_term = pow(x0[p], n - k - dx);
          for (std::size_t i = n - k; i > n - k - dx; --i)
            x_term *= i;
          value += factorials[k] * x_term;
        }
      }
      else
      {
        for (std::size_t k = 0; k + dx <= n; ++k)
        {
          T x_term = pow(1.0 - x0[p], n - k - dx);
          for (std::size_t i = n - k; i > n - k - dx; --i)
            x_term *= -i;
          value += factorials[k] * x_term;
        }
      }
      for (std::size_t dy = 0; dy <= nderiv - dx; ++dy)
        for (std::size_t jy = 0; jy < 2 * n + 1; ++jy)
          P(idx(dx, dy), quad_idx(0, jy), p) = value;
    }
  }

  for (std::size_t j = 0; j < n; ++j)
  {
    for (std::size_t k = 0; k <= j; ++k)
    {
      factorials[k] = (k % 2 == 0 ? 1 : -1)
                      * single_choose(2 * n + 1 - k, j - k)
                      * single_choose(j, k) * pow(2, j - k) * pow(2, n - j)
                      * sqrt(4 * (n - j) + 2);
    }
    for (std::size_t dx = 0; dx <= nderiv; ++dx)
    {
      for (std::size_t p = 0; p < P.extent(2); ++p)
      {
        if (x0[p] <= 0.5)
        {
          T value = 0.0;
          for (std::size_t k = 0; k + dx <= j; ++k)
          {
            T x_term = pow(x0[p], j - k - dx);
            for (std::size_t i = j - k; i > j - k - dx; --i)
              x_term *= i;
            value += factorials[k] * x_term;
          }
          value *= pow(0.5 - x0[p], n - j - dx);
          for (std::size_t i = n - j; i > n - j - dx; --i)
            value *= -i;
          for (std::size_t dy = 0; dy <= nderiv - dx; ++dy)
            for (std::size_t jy = 0; jy < 2 * n + 1; ++jy)
              P(idx(dx, dy), quad_idx(j + 1, jy), p) = value;
        }
        else
        {
          T value = 0.0;
          for (std::size_t k = 0; k + dx <= j; ++k)
          {
            T x_term = pow(1.0 - x0[p], j - k - dx);
            for (std::size_t i = j - k; i > j - k - dx; --i)
              x_term *= -i;
            value += factorials[k] * x_term;
          }
          value *= pow(x0[p] - 0.5, n - j - dx);
          for (std::size_t i = n - j; i > n - j - dx; --i)
            value *= i;
          for (std::size_t dy = 0; dy <= nderiv - dx; ++dy)
            for (std::size_t jy = 0; jy < 2 * n + 1; ++jy)
              P(idx(dx, dy), quad_idx(j + n + 1, jy), p) = value;
        }
      }
    }
  }

  // Multiply by values of polynomials in y
  for (std::size_t k = 0; k <= n; ++k)
  {
    factorials[k] = (k % 2 == 0 ? 1 : -1) * single_choose(2 * n + 1 - k, n - k)
                    * single_choose(n, k) * pow(2, n - k);
  }
  for (std::size_t dy = 0; dy <= nderiv; ++dy)
  {
    for (std::size_t p = 0; p < P.extent(2); ++p)
    {
      T value = 0.0;
      if (x1[p] <= 0.5)
      {
        for (std::size_t k = 0; k + dy <= n; ++k)
        {
          T y_term = pow(x1[p], n - k - dy);
          for (std::size_t i = n - k; i > n - k - dy; --i)
            y_term *= i;
          value += factorials[k] * y_term;
        }
      }
      else
      {
        for (std::size_t k = 0; k + dy <= n; ++k)
        {
          T y_term = pow(1.0 - x1[p], n - k - dy);
          for (std::size_t i = n - k; i > n - k - dy; --i)
            y_term *= -i;
          value += factorials[k] * y_term;
        }
      }
      for (std::size_t dx = 0; dx <= nderiv - dy; ++dx)
        for (std::size_t jx = 0; jx < 2 * n + 1; ++jx)
          P(idx(dx, dy), quad_idx(jx, 0), p) *= value;
    }
  }

  for (std::size_t j = 0; j < n; ++j)
  {
    for (std::size_t k = 0; k <= j; ++k)
    {
      factorials[k] = (k % 2 == 0 ? 1 : -1)
                      * single_choose(2 * n + 1 - k, j - k)
                      * single_choose(j, k) * pow(2, j - k) * pow(2, n - j)
                      * sqrt(4 * (n - j) + 2);
    }
    for (std::size_t dy = 0; dy <= nderiv; ++dy)
    {
      for (std::size_t p = 0; p < P.extent(2); ++p)
      {
        if (x1[p] <= 0.5)
        {
          T value = 0.0;
          for (std::size_t k = 0; k + dy <= j; ++k)
          {
            T y_term = pow(x1[p], j - k - dy);
            for (std::size_t i = j - k; i > j - k - dy; --i)
              y_term *= i;
            value += factorials[k] * y_term;
          }
          value *= pow(0.5 - x1[p], n - j - dy);
          for (std::size_t i = n - j; i > n - j - dy; --i)
            value *= -i;
          for (std::size_t dx = 0; dx <= nderiv - dy; ++dx)
          {
            for (std::size_t jx = 0; jx < 2 * n + 1; ++jx)
            {
              P(idx(dx, dy), quad_idx(jx, j + 1), p) *= value;
              P(idx(dx, dy), quad_idx(jx, j + n + 1), p) *= 0.0;
            }
          }
        }
        else
        {
          T value = 0.0;
          for (std::size_t k = 0; k + dy <= j; ++k)
          {
            T y_term = pow(1.0 - x1[p], j - k - dy);
            for (std::size_t i = j - k; i > j - k - dy; --i)
              y_term *= -i;
            value += factorials[k] * y_term;
          }
          value *= pow(x1[p] - 0.5, n - j - dy);
          for (std::size_t i = n - j; i > n - j - dy; --i)
            value *= i;
          for (std::size_t dx = 0; dx <= nderiv - dy; ++dx)
          {
            for (std::size_t jx = 0; jx < 2 * n + 1; ++jx)
            {
              P(idx(dx, dy), quad_idx(jx, j + 1), p) *= 0.0;
              P(idx(dx, dy), quad_idx(jx, j + n + 1), p) *= value;
            }
          }
        }
      }
    }
  }
}

//-----------------------------------------------------------------------------
/// Compute the complete set of derivatives from 0 to nderiv, for all
/// the piecewise polynomials up to order n on a triangle split into 4 by
/// splitting each edge into two parts
template <typename T>
void tabulate_polyset_triangle_macroedge_derivs(
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 3>>
        P,
    std::size_t n, std::size_t nderiv,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        x)
{
  assert(x.extent(0) > 0);
  assert(P.extent(0) == (nderiv + 1) * (nderiv + 2) / 2);
  assert(P.extent(1) == (n + 1) * (2 * n + 1));
  assert(P.extent(2) == x.extent(0));

  auto x0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
  auto x1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 1);

  std::fill(P.data_handle(), P.data_handle() + P.size(), 0.0);

  if (n == 0)
  {
    for (std::size_t p = 0; p < P.extent(2); ++p)
      P(idx(0, 0), 0, p) = std::sqrt(2);
  }
  else if (n == 1)
  {
    for (std::size_t p = 0; p < P.extent(2); ++p)
    {
      if (x0[p] + x1[p] < 0.5)
      {
        P(idx(0, 0), 0, p) = 6.928203230275509 - 13.856406460551018 * x1[p]
                             - 13.856406460551018 * x0[p];
        P(idx(0, 0), 3, p) = -2.1908902300206643 + 4.381780460041329 * x1[p]
                             + 13.145341380123988 * x0[p];
        P(idx(0, 0), 4, p) = -1.6076739049370008 + 12.402055838085435 * x1[p]
                             + 0.45933540141057166 * x0[p];
        P(idx(0, 0), 5, p) = 1.0894095588038444 - 4.3576382352153775 * x1[p]
                             - 4.3576382352153775 * x0[p];
        if (nderiv >= 1)
        {
          P(idx(0, 1), 0, p) = -13.856406460551018;
          P(idx(0, 1), 3, p) = 4.381780460041329;
          P(idx(0, 1), 4, p) = 12.402055838085435;
          P(idx(0, 1), 5, p) = -4.3576382352153775;
          P(idx(1, 0), 0, p) = -13.856406460551018;
          P(idx(1, 0), 3, p) = 13.145341380123988;
          P(idx(1, 0), 4, p) = 0.45933540141057166;
          P(idx(1, 0), 5, p) = -4.3576382352153775;
        }
      }
      else if (x0[p] > 0.5)
      {
        P(idx(0, 0), 1, p) = -6.928203230275509 + 13.856406460551018 * x0[p];
        P(idx(0, 0), 3, p) = 10.954451150103322 - 8.763560920082657 * x1[p]
                             - 13.145341380123988 * x0[p];
        P(idx(0, 0), 4, p) = -3.4450155105792875 + 2.75601240846343 * x1[p]
                             + 4.134018612695145 * x0[p];
        P(idx(0, 0), 5, p) = -0.36313651960128146 + 11.620368627241007 * x1[p]
                             - 1.4525460784051258 * x0[p];
        if (nderiv >= 1)
        {
          P(idx(0, 1), 3, p) = -8.763560920082657;
          P(idx(0, 1), 4, p) = 2.75601240846343;
          P(idx(0, 1), 5, p) = 11.620368627241007;
          P(idx(1, 0), 1, p) = 13.856406460551018;
          P(idx(1, 0), 3, p) = -13.145341380123988;
          P(idx(1, 0), 4, p) = 4.134018612695145;
          P(idx(1, 0), 5, p) = -1.4525460784051258;
        }
      }
      else if (x1[p] > 0.5)
      {
        P(idx(0, 0), 2, p) = -6.928203230275509 + 13.856406460551018 * x1[p];
        P(idx(0, 0), 4, p) = 11.483385035264291 - 13.78006204231715 * x1[p]
                             - 9.186708028211433 * x0[p];
        P(idx(0, 0), 5, p) = -0.36313651960128146 - 1.4525460784051258 * x1[p]
                             + 11.620368627241007 * x0[p];
        if (nderiv >= 1)
        {
          P(idx(0, 1), 2, p) = 13.856406460551018;
          P(idx(0, 1), 4, p) = -13.78006204231715;
          P(idx(0, 1), 5, p) = -1.4525460784051258;
          P(idx(1, 0), 4, p) = -9.186708028211433;
          P(idx(1, 0), 5, p) = 11.620368627241007;
        }
      }
      else
      {
        P(idx(0, 0), 3, p) = 4.381780460041329 - 8.763560920082657 * x1[p];
        P(idx(0, 0), 4, p) = 3.2153478098740016 + 2.75601240846343 * x1[p]
                             - 9.186708028211433 * x0[p];
        P(idx(0, 0), 5, p) = -6.899593872424347 + 11.620368627241007 * x1[p]
                             + 11.620368627241007 * x0[p];
        if (nderiv >= 1)
        {
          P(idx(0, 1), 3, p) = -8.763560920082657;
          P(idx(0, 1), 4, p) = 2.75601240846343;
          P(idx(0, 1), 5, p) = 11.620368627241007;
          P(idx(1, 0), 4, p) = -9.186708028211433;
          P(idx(1, 0), 5, p) = 11.620368627241007;
        }
      }
    }
  }
  else if (n == 2)
  {
    for (std::size_t p = 0; p < P.extent(2); ++p)
    {
      if (x0[p] + x1[p] < 0.5)
      {
        P(idx(0, 0), 0, p) = 6.928203230275509 - 13.856406460551018 * x1[p]
                             - 13.856406460551018 * x0[p];
        P(idx(0, 0), 3, p) = -2.1908902300206643 + 4.381780460041329 * x1[p]
                             + 13.145341380123988 * x0[p];
        P(idx(0, 0), 4, p) = -1.6076739049370008 + 12.402055838085435 * x1[p]
                             + 0.45933540141057166 * x0[p];
        P(idx(0, 0), 5, p) = 1.0894095588038444 - 4.3576382352153775 * x1[p]
                             - 4.3576382352153775 * x0[p];
        P(idx(0, 0), 6, p) = -8.503281219963945 + 18.006948465806 * x1[p]
                             + 106.04091874307977 * x0[p]
                             - 186.73872483058074 * x0[p] * x1[p]
                             - 186.73872483058074 * x0[p] * x0[p];
        P(idx(0, 0), 7, p) = -11.025314622788574 + 114.36807095525512 * x1[p]
                             + 46.47181295366057 * x0[p]
                             - 193.07346093997572 * x1[p] * x1[p]
                             - 242.1245564220236 * x0[p] * x1[p]
                             - 49.051095482047884 * x0[p] * x0[p];
        P(idx(0, 0), 8, p) = 0.6108718451294817 - 6.108718451294817 * x1[p]
                             - 6.108718451294817 * x0[p]
                             + 12.217436902589634 * x1[p] * x1[p]
                             + 24.43487380517927 * x0[p] * x1[p]
                             + 12.217436902589634 * x0[p] * x0[p];
        P(idx(0, 0), 9, p) = -0.9146023313861029 + 2.3428237060136627 * x1[p]
                             + 15.949222921708396 * x0[p]
                             + 2.117552195820041 * x1[p] * x1[p]
                             - 36.58409325544412 * x0[p] * x1[p]
                             - 38.701645451264156 * x0[p] * x0[p];
        P(idx(0, 0), 10, p) = 0.7688034710261703 - 7.643595203265971 * x1[p]
                              - 7.732474217257435 * x0[p]
                              + 15.242750899536208 * x1[p] * x1[p]
                              + 30.75213884104681 * x0[p] * x1[p]
                              + 15.509387941510603 * x0[p] * x0[p];
        P(idx(0, 0), 11, p) = -0.8466863237791316 + 15.856125699863737 * x1[p]
                              + 1.0776007757188948 * x0[p]
                              - 39.101513861799894 * x1[p] * x1[p]
                              - 33.86745295116526 * x0[p] * x1[p]
                              + 5.234060910634632 * x0[p] * x0[p];
        P(idx(0, 0), 12, p) = 1.2766744019646143 - 12.766744019646143 * x1[p]
                              - 12.766744019646143 * x0[p]
                              + 0.785645785824378 * x1[p] * x1[p]
                              + 150.05834509245622 * x0[p] * x1[p]
                              + 0.785645785824378 * x0[p] * x0[p];
        P(idx(0, 0), 13, p) = -1.4902171140068574 + 1.7816172183668608 * x1[p]
                              + 28.022725061770288 * x0[p]
                              + 5.689096337971468 * x1[p] * x1[p]
                              - 44.1357919722882 * x0[p] * x1[p]
                              - 73.03422719223882 * x0[p] * x0[p];
        P(idx(0, 0), 14, p) = -1.7925313955666775 + 28.833058192519324 * x1[p]
                              + 7.017569718814227 * x0[p]
                              - 73.22681445719194 * x1[p] * x1[p]
                              - 53.08944048146415 * x0[p] * x1[p]
                              - 7.780349036076643 * x0[p] * x0[p];
        if (nderiv >= 1)
        {
          P(idx(0, 1), 0, p) = -13.856406460551018;
          P(idx(0, 1), 3, p) = 4.381780460041329;
          P(idx(0, 1), 4, p) = 12.402055838085435;
          P(idx(0, 1), 5, p) = -4.3576382352153775;
          P(idx(0, 1), 6, p) = 18.006948465806 - 186.73872483058074 * x0[p];
          P(idx(0, 1), 7, p) = 114.36807095525512 - 386.14692187995144 * x1[p]
                               - 242.1245564220236 * x0[p];
          P(idx(0, 1), 8, p) = -6.108718451294817 + 24.43487380517927 * x1[p]
                               + 24.43487380517927 * x0[p];
          P(idx(0, 1), 9, p) = 2.3428237060136627 + 4.235104391640082 * x1[p]
                               - 36.58409325544412 * x0[p];
          P(idx(0, 1), 10, p) = -7.643595203265971 + 30.485501799072416 * x1[p]
                                + 30.75213884104681 * x0[p];
          P(idx(0, 1), 11, p) = 15.856125699863737 - 78.20302772359979 * x1[p]
                                - 33.86745295116526 * x0[p];
          P(idx(0, 1), 12, p) = -12.766744019646143 + 1.571291571648756 * x1[p]
                                + 150.05834509245622 * x0[p];
          P(idx(0, 1), 13, p) = 1.7816172183668608 + 11.378192675942936 * x1[p]
                                - 44.1357919722882 * x0[p];
          P(idx(0, 1), 14, p) = 28.833058192519324 - 146.45362891438387 * x1[p]
                                - 53.08944048146415 * x0[p];
          P(idx(1, 0), 0, p) = -13.856406460551018;
          P(idx(1, 0), 3, p) = 13.145341380123988;
          P(idx(1, 0), 4, p) = 0.45933540141057166;
          P(idx(1, 0), 5, p) = -4.3576382352153775;
          P(idx(1, 0), 6, p) = 106.04091874307977 - 186.73872483058074 * x1[p]
                               - 373.4774496611615 * x0[p];
          P(idx(1, 0), 7, p) = 46.47181295366057 - 242.1245564220236 * x1[p]
                               - 98.10219096409577 * x0[p];
          P(idx(1, 0), 8, p) = -6.108718451294817 + 24.43487380517927 * x1[p]
                               + 24.43487380517927 * x0[p];
          P(idx(1, 0), 9, p) = 15.949222921708396 - 36.58409325544412 * x1[p]
                               - 77.40329090252831 * x0[p];
          P(idx(1, 0), 10, p) = -7.732474217257435 + 30.75213884104681 * x1[p]
                                + 31.018775883021206 * x0[p];
          P(idx(1, 0), 11, p) = 1.0776007757188948 - 33.86745295116526 * x1[p]
                                + 10.468121821269264 * x0[p];
          P(idx(1, 0), 12, p) = -12.766744019646143 + 150.05834509245622 * x1[p]
                                + 1.571291571648756 * x0[p];
          P(idx(1, 0), 13, p) = 28.022725061770288 - 44.1357919722882 * x1[p]
                                - 146.06845438447763 * x0[p];
          P(idx(1, 0), 14, p) = 7.017569718814227 - 53.08944048146415 * x1[p]
                                - 15.560698072153286 * x0[p];
        }
        if (nderiv >= 2)
        {
          P(idx(0, 2), 7, p) = -386.14692187995144;
          P(idx(0, 2), 8, p) = 24.43487380517927;
          P(idx(0, 2), 9, p) = 4.235104391640082;
          P(idx(0, 2), 10, p) = 30.485501799072416;
          P(idx(0, 2), 11, p) = -78.20302772359979;
          P(idx(0, 2), 12, p) = 1.571291571648756;
          P(idx(0, 2), 13, p) = 11.378192675942936;
          P(idx(0, 2), 14, p) = -146.45362891438387;
          P(idx(1, 1), 6, p) = -186.73872483058074;
          P(idx(1, 1), 7, p) = -242.1245564220236;
          P(idx(1, 1), 8, p) = 24.43487380517927;
          P(idx(1, 1), 9, p) = -36.58409325544412;
          P(idx(1, 1), 10, p) = 30.75213884104681;
          P(idx(1, 1), 11, p) = -33.86745295116526;
          P(idx(1, 1), 12, p) = 150.05834509245622;
          P(idx(1, 1), 13, p) = -44.1357919722882;
          P(idx(1, 1), 14, p) = -53.08944048146415;
          P(idx(2, 0), 6, p) = -373.4774496611615;
          P(idx(2, 0), 7, p) = -98.10219096409577;
          P(idx(2, 0), 8, p) = 24.43487380517927;
          P(idx(2, 0), 9, p) = -77.40329090252831;
          P(idx(2, 0), 10, p) = 31.018775883021206;
          P(idx(2, 0), 11, p) = 10.468121821269264;
          P(idx(2, 0), 12, p) = 1.571291571648756;
          P(idx(2, 0), 13, p) = -146.06845438447763;
          P(idx(2, 0), 14, p) = -15.560698072153286;
        }
      }
      else if (x0[p] > 0.5)
      {
        P(idx(0, 0), 1, p) = -6.928203230275509 + 13.856406460551018 * x0[p];
        P(idx(0, 0), 3, p) = 10.954451150103322 - 8.763560920082657 * x1[p]
                             - 13.145341380123988 * x0[p];
        P(idx(0, 0), 4, p) = -3.4450155105792875 + 2.75601240846343 * x1[p]
                             + 4.134018612695145 * x0[p];
        P(idx(0, 0), 5, p) = -0.36313651960128146 + 11.620368627241007 * x1[p]
                             - 1.4525460784051258 * x0[p];
        P(idx(0, 0), 6, p) = -5.168661133703574 + 5.3353921380165925 * x1[p]
                             + 6.002316155268667 * x0[p];
        P(idx(0, 0), 7, p) = 0.19381891831812234 + 1.4014598709156538 * x1[p]
                             - 0.4920018695767721 * x0[p];
        P(idx(0, 0), 8, p) = 9.773949522071707 - 99.36848680772903 * x1[p]
                             - 18.326155353884452 * x0[p]
                             + 187.3340325063744 * x0[p] * x1[p];
        P(idx(0, 0), 9, p) = -92.87043279242236 + 79.83622321261943 * x1[p]
                             + 279.1564554319356 * x0[p]
                             - 148.94952254002249 * x0[p] * x1[p]
                             - 197.29278862757363 * x0[p] * x0[p];
        P(idx(0, 0), 10, p) = 3.2129763557914517 - 28.619042505251656 * x1[p]
                              - 5.510498867470816 * x0[p]
                              + 42.928563757877484 * x0[p] * x1[p]
                              + 1.2887457028762392 * x0[p] * x0[p];
        P(idx(0, 0), 11, p) = 9.621435497490133 - 7.389262462072421 * x1[p]
                              - 25.400589713373947 * x0[p]
                              + 11.083893693108632 * x0[p] * x1[p]
                              + 16.317954603743264 * x0[p] * x0[p];
        P(idx(0, 0), 12, p) = -44.29078117584932 + 51.85262186440895 * x1[p]
                              + 114.9006961768153 * x0[p]
                              - 77.77893279661343 * x0[p] * x1[p]
                              - 72.27941229584279 * x0[p] * x0[p];
        P(idx(0, 0), 13, p) = -17.757288224756483 + 158.3948297834158 * x1[p]
                              + 29.290994946031443 * x0[p]
                              - 150.2899812849469 * x1[p] * x1[p]
                              - 162.44725403265025 * x0[p] * x1[p]
                              - 10.502482517762614 * x0[p] * x0[p];
        P(idx(0, 0), 14, p) = -2.5553107128290935 - 15.713253935605769 * x1[p]
                              + 8.543128353339059 * x0[p]
                              - 27.917723011804423 * x1[p] * x1[p]
                              + 37.528742409310865 * x0[p] * x1[p]
                              - 7.780349036076643 * x0[p] * x0[p];
        if (nderiv >= 1)
        {
          P(idx(0, 1), 3, p) = -8.763560920082657;
          P(idx(0, 1), 4, p) = 2.75601240846343;
          P(idx(0, 1), 5, p) = 11.620368627241007;
          P(idx(0, 1), 6, p) = 5.3353921380165925;
          P(idx(0, 1), 7, p) = 1.4014598709156538;
          P(idx(0, 1), 8, p) = -99.36848680772903 + 187.3340325063744 * x0[p];
          P(idx(0, 1), 9, p) = 79.83622321261943 - 148.94952254002249 * x0[p];
          P(idx(0, 1), 10, p)
              = -28.619042505251656 + 42.928563757877484 * x0[p];
          P(idx(0, 1), 11, p) = -7.389262462072421 + 11.083893693108632 * x0[p];
          P(idx(0, 1), 12, p) = 51.85262186440895 - 77.77893279661343 * x0[p];
          P(idx(0, 1), 13, p) = 158.3948297834158 - 300.5799625698938 * x1[p]
                                - 162.44725403265025 * x0[p];
          P(idx(0, 1), 14, p) = -15.713253935605769 - 55.835446023608846 * x1[p]
                                + 37.528742409310865 * x0[p];
          P(idx(1, 0), 1, p) = 13.856406460551018;
          P(idx(1, 0), 3, p) = -13.145341380123988;
          P(idx(1, 0), 4, p) = 4.134018612695145;
          P(idx(1, 0), 5, p) = -1.4525460784051258;
          P(idx(1, 0), 6, p) = 6.002316155268667;
          P(idx(1, 0), 7, p) = -0.4920018695767721;
          P(idx(1, 0), 8, p) = -18.326155353884452 + 187.3340325063744 * x1[p];
          P(idx(1, 0), 9, p) = 279.1564554319356 - 148.94952254002249 * x1[p]
                               - 394.58557725514726 * x0[p];
          P(idx(1, 0), 10, p) = -5.510498867470816 + 42.928563757877484 * x1[p]
                                + 2.5774914057524785 * x0[p];
          P(idx(1, 0), 11, p) = -25.400589713373947 + 11.083893693108632 * x1[p]
                                + 32.63590920748653 * x0[p];
          P(idx(1, 0), 12, p) = 114.9006961768153 - 77.77893279661343 * x1[p]
                                - 144.55882459168558 * x0[p];
          P(idx(1, 0), 13, p) = 29.290994946031443 - 162.44725403265025 * x1[p]
                                - 21.004965035525228 * x0[p];
          P(idx(1, 0), 14, p) = 8.543128353339059 + 37.528742409310865 * x1[p]
                                - 15.560698072153286 * x0[p];
        }
        if (nderiv >= 2)
        {
          P(idx(0, 2), 13, p) = -300.5799625698938;
          P(idx(0, 2), 14, p) = -55.835446023608846;
          P(idx(1, 1), 8, p) = 187.3340325063744;
          P(idx(1, 1), 9, p) = -148.94952254002249;
          P(idx(1, 1), 10, p) = 42.928563757877484;
          P(idx(1, 1), 11, p) = 11.083893693108632;
          P(idx(1, 1), 12, p) = -77.77893279661343;
          P(idx(1, 1), 13, p) = -162.44725403265025;
          P(idx(1, 1), 14, p) = 37.528742409310865;
          P(idx(2, 0), 9, p) = -394.58557725514726;
          P(idx(2, 0), 10, p) = 2.5774914057524785;
          P(idx(2, 0), 11, p) = 32.63590920748653;
          P(idx(2, 0), 12, p) = -144.55882459168558;
          P(idx(2, 0), 13, p) = -21.004965035525228;
          P(idx(2, 0), 14, p) = -15.560698072153286;
        }
      }
      else if (x1[p] > 0.5)
      {
        P(idx(0, 0), 2, p) = -6.928203230275509 + 13.856406460551018 * x1[p];
        P(idx(0, 0), 4, p) = 11.483385035264291 - 13.78006204231715 * x1[p]
                             - 9.186708028211433 * x0[p];
        P(idx(0, 0), 5, p) = -0.36313651960128146 - 1.4525460784051258 * x1[p]
                             + 11.620368627241007 * x0[p];
        P(idx(0, 0), 6, p) = 1.5005790388171667 - 2.000772051756222 * x1[p];
        P(idx(0, 0), 7, p) = -4.949836990893586 + 5.680385221477278 * x1[p]
                             + 5.51638459828502 * x0[p];
        P(idx(0, 0), 8, p) = 0.40724789675298784 + 0.40724789675298784 * x1[p]
                             - 5.701470554541829 * x0[p];
        P(idx(0, 0), 9, p) = 1.9981582954174217 - 2.423921449683366 * x1[p]
                             - 1.441737665239177 * x0[p];
        P(idx(0, 0), 10, p) = 10.094434014080612 - 18.67348083960675 * x1[p]
                              - 103.09965623009914 * x0[p]
                              + 191.9786702215639 * x0[p] * x1[p];
        P(idx(0, 0), 11, p) = -93.36641006764424 + 280.33014465487247 * x1[p]
                              + 80.35822927503759 * x0[p]
                              - 197.97065679635696 * x1[p] * x1[p]
                              - 149.63256485696652 * x0[p] * x1[p];
        P(idx(0, 0), 12, p) = -44.29078117584932 + 114.9006961768153 * x1[p]
                              + 51.85262186440895 * x0[p]
                              - 72.27941229584279 * x1[p] * x1[p]
                              - 77.77893279661343 * x0[p] * x1[p];
        P(idx(0, 0), 13, p) = 0.7307650285504752 + 3.0498871026280163 * x1[p]
                              - 44.37736718833795 * x0[p]
                              - 5.731372000780173 * x1[p] * x1[p]
                              + 66.56605078250692 * x0[p] * x1[p];
        P(idx(0, 0), 14, p) = -17.925313955666773 + 30.358616827044155 * x1[p]
                              + 152.86097517938816 * x0[p]
                              - 11.746801485841205 * x1[p] * x1[p]
                              - 152.86097517938816 * x0[p] * x1[p]
                              - 152.86097517938816 * x0[p] * x0[p];
        if (nderiv >= 1)
        {
          P(idx(0, 1), 2, p) = 13.856406460551018;
          P(idx(0, 1), 4, p) = -13.78006204231715;
          P(idx(0, 1), 5, p) = -1.4525460784051258;
          P(idx(0, 1), 6, p) = -2.000772051756222;
          P(idx(0, 1), 7, p) = 5.680385221477278;
          P(idx(0, 1), 8, p) = 0.40724789675298784;
          P(idx(0, 1), 9, p) = -2.423921449683366;
          P(idx(0, 1), 10, p) = -18.67348083960675 + 191.9786702215639 * x0[p];
          P(idx(0, 1), 11, p) = 280.33014465487247 - 395.9413135927139 * x1[p]
                                - 149.63256485696652 * x0[p];
          P(idx(0, 1), 12, p) = 114.9006961768153 - 144.55882459168558 * x1[p]
                                - 77.77893279661343 * x0[p];
          P(idx(0, 1), 13, p) = 3.0498871026280163 - 11.462744001560345 * x1[p]
                                + 66.56605078250692 * x0[p];
          P(idx(0, 1), 14, p) = 30.358616827044155 - 23.49360297168241 * x1[p]
                                - 152.86097517938816 * x0[p];
          P(idx(1, 0), 4, p) = -9.186708028211433;
          P(idx(1, 0), 5, p) = 11.620368627241007;
          P(idx(1, 0), 7, p) = 5.51638459828502;
          P(idx(1, 0), 8, p) = -5.701470554541829;
          P(idx(1, 0), 9, p) = -1.441737665239177;
          P(idx(1, 0), 10, p) = -103.09965623009914 + 191.9786702215639 * x1[p];
          P(idx(1, 0), 11, p) = 80.35822927503759 - 149.63256485696652 * x1[p];
          P(idx(1, 0), 12, p) = 51.85262186440895 - 77.77893279661343 * x1[p];
          P(idx(1, 0), 13, p) = -44.37736718833795 + 66.56605078250692 * x1[p];
          P(idx(1, 0), 14, p) = 152.86097517938816 - 152.86097517938816 * x1[p]
                                - 305.7219503587763 * x0[p];
        }
        if (nderiv >= 2)
        {
          P(idx(0, 2), 11, p) = -395.9413135927139;
          P(idx(0, 2), 12, p) = -144.55882459168558;
          P(idx(0, 2), 13, p) = -11.462744001560345;
          P(idx(0, 2), 14, p) = -23.49360297168241;
          P(idx(1, 1), 10, p) = 191.9786702215639;
          P(idx(1, 1), 11, p) = -149.63256485696652;
          P(idx(1, 1), 12, p) = -77.77893279661343;
          P(idx(1, 1), 13, p) = 66.56605078250692;
          P(idx(1, 1), 14, p) = -152.86097517938816;
          P(idx(2, 0), 14, p) = -305.7219503587763;
        }
      }
      else
      {
        P(idx(0, 0), 3, p) = 4.381780460041329 - 8.763560920082657 * x1[p];
        P(idx(0, 0), 4, p) = 3.2153478098740016 + 2.75601240846343 * x1[p]
                             - 9.186708028211433 * x0[p];
        P(idx(0, 0), 5, p) = -6.899593872424347 + 11.620368627241007 * x1[p]
                             + 11.620368627241007 * x0[p];
        P(idx(0, 0), 6, p) = -2.1675030560692408 + 5.3353921380165925 * x1[p];
        P(idx(0, 0), 7, p) = -2.810374315612774 + 1.4014598709156538 * x1[p]
                             + 5.51638459828502 * x0[p];
        P(idx(0, 0), 8, p) = 3.4616071224003964 - 5.701470554541829 * x1[p]
                             - 5.701470554541829 * x0[p];
        P(idx(0, 0), 9, p) = -1.894533400728356 + 5.361461942608189 * x1[p]
                             - 1.441737665239177 * x0[p];
        P(idx(0, 0), 10, p) = 4.335073907433695 - 7.154760626312914 * x1[p]
                              - 7.110321119317182 * x0[p];
        P(idx(0, 0), 11, p) = -1.7703441315381843 - 1.8473156155181052 * x1[p]
                              + 5.541946846554316 * x0[p];
        P(idx(0, 0), 12, p) = 25.729899485748383 - 61.28037129430149 * x1[p]
                              - 61.28037129430149 * x0[p]
                              + 148.48705352080745 * x0[p] * x1[p];
        P(idx(0, 0), 13, p) = -31.96040108338111 + 140.71152396857454 * x1[p]
                              + 52.44597940439939 * x0[p]
                              - 150.2899812849469 * x1[p] * x1[p]
                              - 127.08064240296775 * x0[p] * x1[p];
        P(idx(0, 0), 14, p) = -38.44407759002576 + 79.48160485874374 * x1[p]
                              + 152.86097517938816 * x0[p]
                              - 27.917723011804423 * x1[p] * x1[p]
                              - 152.86097517938816 * x0[p] * x1[p]
                              - 152.86097517938816 * x0[p] * x0[p];
        if (nderiv >= 1)
        {
          P(idx(0, 1), 3, p) = -8.763560920082657;
          P(idx(0, 1), 4, p) = 2.75601240846343;
          P(idx(0, 1), 5, p) = 11.620368627241007;
          P(idx(0, 1), 6, p) = 5.3353921380165925;
          P(idx(0, 1), 7, p) = 1.4014598709156538;
          P(idx(0, 1), 8, p) = -5.701470554541829;
          P(idx(0, 1), 9, p) = 5.361461942608189;
          P(idx(0, 1), 10, p) = -7.154760626312914;
          P(idx(0, 1), 11, p) = -1.8473156155181052;
          P(idx(0, 1), 12, p) = -61.28037129430149 + 148.48705352080745 * x0[p];
          P(idx(0, 1), 13, p) = 140.71152396857454 - 300.5799625698938 * x1[p]
                                - 127.08064240296775 * x0[p];
          P(idx(0, 1), 14, p) = 79.48160485874374 - 55.835446023608846 * x1[p]
                                - 152.86097517938816 * x0[p];
          P(idx(1, 0), 4, p) = -9.186708028211433;
          P(idx(1, 0), 5, p) = 11.620368627241007;
          P(idx(1, 0), 7, p) = 5.51638459828502;
          P(idx(1, 0), 8, p) = -5.701470554541829;
          P(idx(1, 0), 9, p) = -1.441737665239177;
          P(idx(1, 0), 10, p) = -7.110321119317182;
          P(idx(1, 0), 11, p) = 5.541946846554316;
          P(idx(1, 0), 12, p) = -61.28037129430149 + 148.48705352080745 * x1[p];
          P(idx(1, 0), 13, p) = 52.44597940439939 - 127.08064240296775 * x1[p];
          P(idx(1, 0), 14, p) = 152.86097517938816 - 152.86097517938816 * x1[p]
                                - 305.7219503587763 * x0[p];
        }
        if (nderiv >= 2)
        {
          P(idx(0, 2), 13, p) = -300.5799625698938;
          P(idx(0, 2), 14, p) = -55.835446023608846;
          P(idx(1, 1), 12, p) = 148.48705352080745;
          P(idx(1, 1), 13, p) = -127.08064240296775;
          P(idx(1, 1), 14, p) = -152.86097517938816;
          P(idx(2, 0), 14, p) = -305.7219503587763;
        }
      }
    }
  }
  else
  {
    throw std::runtime_error("Only degree 0 to 2 macro polysets are currently "
                             "implemented on a triangle.");
  }
}
//-----------------------------------------------------------------------------
/// Compute the complete set of derivatives from 0 to nderiv, for all
/// the piecewise polynomials up to order n on a tetrahedron split into 8 by
/// splitting each edge into two parts
template <typename T>
void tabulate_polyset_tetrahedron_macroedge_derivs(
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 3>>
        P,
    std::size_t n, std::size_t nderiv,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        x)
{
  assert(x.extent(0) > 0);
  assert(P.extent(0) == (nderiv + 1) * (nderiv + 2) * (nderiv + 3) / 6);
  assert(P.extent(1) == (n + 1) * (2 * n + 1) * (2 * n + 3) / 3);
  assert(P.extent(2) == x.extent(0));

  auto x0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
  auto x1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 1);
  auto x2 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 2);

  std::fill(P.data_handle(), P.data_handle() + P.size(), 0.0);

  if (n == 0)
  {
    for (std::size_t p = 0; p < P.extent(2); ++p)
      P(idx(0, 0), 0, p) = std::sqrt(6);
  }
  else if (n == 1)
  {
    for (std::size_t p = 0; p < P.extent(2); ++p)
    {
      if (x0[p] + x1[p] + x2[p] < 0.5)
      {
        P(idx(0, 0), 0, p) = 21.908902300206645 - 43.81780460041329 * x2[p]
                             - 43.81780460041329 * x1[p]
                             - 43.81780460041329 * x0[p];
        P(idx(0, 0), 4, p) = -5.855400437691199 + 11.710800875382398 * x2[p]
                             + 11.710800875382398 * x1[p]
                             + 35.1324026261472 * x0[p];
        P(idx(0, 0), 5, p) = -3.1326068447244277 + 6.2652136894488555 * x2[p]
                             + 25.75698961217863 * x1[p]
                             - 0.6961348543832062 * x0[p];
        P(idx(0, 0), 6, p) = -4.079436335011508 + 32.853642355761124 * x2[p]
                             + 3.359535805303594 * x1[p]
                             + 4.581185189050355 * x0[p];
        P(idx(0, 0), 7, p) = 1.6490105505038288 - 7.300270809207225 * x2[p]
                             - 8.666892660787566 * x1[p]
                             - 0.5229420350434975 * x0[p];
        P(idx(0, 0), 8, p) = 3.365373344712382 - 10.492441445989503 * x2[p]
                             - 11.258215021433038 * x1[p]
                             - 11.903076979701279 * x0[p];
        P(idx(0, 0), 9, p) = 0.8017837257372732 + 2.6726124191242437 * x2[p]
                             - 5.3452248382484875 * x1[p]
                             - 5.3452248382484875 * x0[p];
        if (nderiv >= 1)
        {
          P(idx(0, 0, 1), 0, p) = -43.81780460041329;
          P(idx(0, 0, 1), 4, p) = 11.710800875382398;
          P(idx(0, 0, 1), 5, p) = 6.2652136894488555;
          P(idx(0, 0, 1), 6, p) = 32.853642355761124;
          P(idx(0, 0, 1), 7, p) = -7.300270809207225;
          P(idx(0, 0, 1), 8, p) = -10.492441445989503;
          P(idx(0, 0, 1), 9, p) = 2.6726124191242437;
          P(idx(0, 1, 0), 0, p) = -43.81780460041329;
          P(idx(0, 1, 0), 4, p) = 11.710800875382398;
          P(idx(0, 1, 0), 5, p) = 25.75698961217863;
          P(idx(0, 1, 0), 6, p) = 3.359535805303594;
          P(idx(0, 1, 0), 7, p) = -8.666892660787566;
          P(idx(0, 1, 0), 8, p) = -11.258215021433038;
          P(idx(0, 1, 0), 9, p) = -5.3452248382484875;
          P(idx(1, 0, 0), 0, p) = -43.81780460041329;
          P(idx(1, 0, 0), 4, p) = 35.1324026261472;
          P(idx(1, 0, 0), 5, p) = -0.6961348543832062;
          P(idx(1, 0, 0), 6, p) = 4.581185189050355;
          P(idx(1, 0, 0), 7, p) = -0.5229420350434975;
          P(idx(1, 0, 0), 8, p) = -11.903076979701279;
          P(idx(1, 0, 0), 9, p) = -5.3452248382484875;
        }
      }
      else if (x0[p] > 0.5)
      {
        P(idx(0, 0), 1, p) = -21.908902300206645 + 43.81780460041329 * x0[p];
        P(idx(0, 0), 4, p) = 29.277002188455995 - 23.421601750764797 * x2[p]
                             - 23.421601750764797 * x1[p]
                             - 35.1324026261472 * x0[p];
        P(idx(0, 0), 5, p) = -8.701685679790078 + 6.961348543832062 * x2[p]
                             + 6.961348543832062 * x1[p]
                             + 10.442022815748093 * x0[p];
        P(idx(0, 0), 6, p) = -4.472109351215823 + 3.5776874809726587 * x2[p]
                             + 3.5776874809726587 * x1[p]
                             + 5.366531221458988 * x0[p];
        P(idx(0, 0), 7, p) = 3.4688488324552 - 2.77507906596416 * x2[p]
                             - 2.77507906596416 * x1[p]
                             - 4.16261859894624 * x0[p];
        P(idx(0, 0), 8, p) = -1.148660363165304 + 26.439340288997876 * x2[p]
                             + 5.172330290276515 * x1[p]
                             - 2.8750095639459072 * x0[p];
        P(idx(0, 0), 9, p) = 0.8017837257372732 + 29.398736610366683 * x1[p]
                             - 5.3452248382484875 * x0[p];
        if (nderiv >= 1)
        {
          P(idx(0, 0, 1), 4, p) = -23.421601750764797;
          P(idx(0, 0, 1), 5, p) = 6.961348543832062;
          P(idx(0, 0, 1), 6, p) = 3.5776874809726587;
          P(idx(0, 0, 1), 7, p) = -2.77507906596416;
          P(idx(0, 0, 1), 8, p) = 26.439340288997876;
          P(idx(0, 1, 0), 4, p) = -23.421601750764797;
          P(idx(0, 1, 0), 5, p) = 6.961348543832062;
          P(idx(0, 1, 0), 6, p) = 3.5776874809726587;
          P(idx(0, 1, 0), 7, p) = -2.77507906596416;
          P(idx(0, 1, 0), 8, p) = 5.172330290276515;
          P(idx(0, 1, 0), 9, p) = 29.398736610366683;
          P(idx(1, 0, 0), 1, p) = 43.81780460041329;
          P(idx(1, 0, 0), 4, p) = -35.1324026261472;
          P(idx(1, 0, 0), 5, p) = 10.442022815748093;
          P(idx(1, 0, 0), 6, p) = 5.366531221458988;
          P(idx(1, 0, 0), 7, p) = -4.16261859894624;
          P(idx(1, 0, 0), 8, p) = -2.8750095639459072;
          P(idx(1, 0, 0), 9, p) = -5.3452248382484875;
        }
      }
      else if (x1[p] > 0.5)
      {
        P(idx(0, 0), 2, p) = -21.908902300206645 + 43.81780460041329 * x1[p];
        P(idx(0, 0), 5, p) = 24.36471990341222 - 19.491775922729772 * x2[p]
                             - 29.237663884094662 * x1[p]
                             - 19.491775922729772 * x0[p];
        P(idx(0, 0), 6, p) = -5.999171080899275 + 4.79933686471942 * x2[p]
                             + 7.199005297079131 * x1[p]
                             + 4.79933686471942 * x0[p];
        P(idx(0, 0), 7, p) = -0.4985380734081343 + 30.219077065046907 * x2[p]
                             - 4.371795412963639 * x1[p]
                             + 5.368871559779907 * x0[p];
        P(idx(0, 0), 8, p) = -6.952417987579472 - 0.6448619582682409 * x2[p]
                             + 9.377367643150668 * x1[p]
                             + 4.527468332008274 * x0[p];
        P(idx(0, 0), 9, p) = 0.8017837257372732 - 5.3452248382484875 * x1[p]
                             + 29.398736610366683 * x0[p];
        if (nderiv >= 1)
        {
          P(idx(0, 0, 1), 5, p) = -19.491775922729772;
          P(idx(0, 0, 1), 6, p) = 4.79933686471942;
          P(idx(0, 0, 1), 7, p) = 30.219077065046907;
          P(idx(0, 0, 1), 8, p) = -0.6448619582682409;
          P(idx(0, 1, 0), 2, p) = 43.81780460041329;
          P(idx(0, 1, 0), 5, p) = -29.237663884094662;
          P(idx(0, 1, 0), 6, p) = 7.199005297079131;
          P(idx(0, 1, 0), 7, p) = -4.371795412963639;
          P(idx(0, 1, 0), 8, p) = 9.377367643150668;
          P(idx(0, 1, 0), 9, p) = -5.3452248382484875;
          P(idx(1, 0, 0), 5, p) = -19.491775922729772;
          P(idx(1, 0, 0), 6, p) = 4.79933686471942;
          P(idx(1, 0, 0), 7, p) = 5.368871559779907;
          P(idx(1, 0, 0), 8, p) = 4.527468332008274;
          P(idx(1, 0, 0), 9, p) = 29.398736610366683;
        }
      }
      else if (x2[p] > 0.5)
      {
        P(idx(0, 0), 3, p) = -21.908902300206645 + 43.81780460041329 * x2[p];
        P(idx(0, 0), 6, p) = 30.868462107172636 - 37.04215452860716 * x2[p]
                             - 24.69476968573811 * x1[p]
                             - 24.69476968573811 * x0[p];
        P(idx(0, 0), 7, p) = 1.209739241067291 - 6.421728190334149 * x2[p]
                             + 28.85245521346657 * x1[p]
                             + 4.002249708199567 * x0[p];
        P(idx(0, 0), 8, p) = -0.6784485185947118 - 2.4047977193753147 * x2[p]
                             - 1.4106355337117769 * x1[p]
                             + 25.0287047552861 * x0[p];
        P(idx(0, 0), 9, p) = 3.474396144861517 - 2.6726124191242437 * x2[p]
                             - 8.017837257372731 * x1[p]
                             - 8.017837257372731 * x0[p];
        if (nderiv >= 1)
        {
          P(idx(0, 0, 1), 3, p) = 43.81780460041329;
          P(idx(0, 0, 1), 6, p) = -37.04215452860716;
          P(idx(0, 0, 1), 7, p) = -6.421728190334149;
          P(idx(0, 0, 1), 8, p) = -2.4047977193753147;
          P(idx(0, 0, 1), 9, p) = -2.6726124191242437;
          P(idx(0, 1, 0), 6, p) = -24.69476968573811;
          P(idx(0, 1, 0), 7, p) = 28.85245521346657;
          P(idx(0, 1, 0), 8, p) = -1.4106355337117769;
          P(idx(0, 1, 0), 9, p) = -8.017837257372731;
          P(idx(1, 0, 0), 6, p) = -24.69476968573811;
          P(idx(1, 0, 0), 7, p) = 4.002249708199567;
          P(idx(1, 0, 0), 8, p) = 25.0287047552861;
          P(idx(1, 0, 0), 9, p) = -8.017837257372731;
        }
      }
      else if (x1[p] + x2[p] < 0.5 && x0[p] + x1[p] < 0.5)
      {
        P(idx(0, 0), 4, p) = 11.710800875382398 - 23.421601750764797 * x2[p]
                             - 23.421601750764797 * x1[p];
        P(idx(0, 0), 5, p) = -3.480674271916031 + 6.961348543832062 * x2[p]
                             + 26.453124466561835 * x1[p];
        P(idx(0, 0), 6, p) = 10.558541102382724 + 3.5776874809726587 * x2[p]
                             - 25.91641906948487 * x1[p]
                             - 24.69476968573811 * x0[p];
        P(idx(0, 0), 7, p) = -0.6135853211177037 - 2.77507906596416 * x2[p]
                             - 4.1417009175445 * x1[p]
                             + 4.002249708199567 * x0[p];
        P(idx(0, 0), 8, p) = -15.100517522781306 + 26.439340288997876 * x2[p]
                             + 25.67356671355434 * x1[p]
                             + 25.0287047552861 * x0[p];
        P(idx(0, 0), 9, p) = 2.138089935299395 - 8.017837257372731 * x1[p]
                             - 8.017837257372731 * x0[p];
        if (nderiv >= 1)
        {
          P(idx(0, 0, 1), 4, p) = -23.421601750764797;
          P(idx(0, 0, 1), 5, p) = 6.961348543832062;
          P(idx(0, 0, 1), 6, p) = 3.5776874809726587;
          P(idx(0, 0, 1), 7, p) = -2.77507906596416;
          P(idx(0, 0, 1), 8, p) = 26.439340288997876;
          P(idx(0, 1, 0), 4, p) = -23.421601750764797;
          P(idx(0, 1, 0), 5, p) = 26.453124466561835;
          P(idx(0, 1, 0), 6, p) = -25.91641906948487;
          P(idx(0, 1, 0), 7, p) = -4.1417009175445;
          P(idx(0, 1, 0), 8, p) = 25.67356671355434;
          P(idx(0, 1, 0), 9, p) = -8.017837257372731;
          P(idx(1, 0, 0), 6, p) = -24.69476968573811;
          P(idx(1, 0, 0), 7, p) = 4.002249708199567;
          P(idx(1, 0, 0), 8, p) = 25.0287047552861;
          P(idx(1, 0, 0), 9, p) = -8.017837257372731;
        }
      }
      else if (x1[p] + x2[p] < 0.5)
      {
        P(idx(0, 0), 4, p) = 11.710800875382398 - 23.421601750764797 * x2[p]
                             - 23.421601750764797 * x1[p];
        P(idx(0, 0), 5, p) = 6.2652136894488555 + 6.961348543832062 * x2[p]
                             + 6.961348543832062 * x1[p]
                             - 19.491775922729772 * x0[p];
        P(idx(0, 0), 6, p) = -4.18851217284604 + 3.5776874809726587 * x2[p]
                             + 3.5776874809726587 * x1[p]
                             + 4.79933686471942 * x0[p];
        P(idx(0, 0), 7, p) = -1.2968962469078738 - 2.77507906596416 * x2[p]
                             - 2.77507906596416 * x1[p]
                             + 5.368871559779907 * x0[p];
        P(idx(0, 0), 8, p) = -4.849899311142394 + 26.439340288997876 * x2[p]
                             + 5.172330290276515 * x1[p]
                             + 4.527468332008274 * x0[p];
        P(idx(0, 0), 9, p) = -16.57019699857031 + 29.398736610366683 * x1[p]
                             + 29.398736610366683 * x0[p];
        if (nderiv >= 1)
        {
          P(idx(0, 0, 1), 4, p) = -23.421601750764797;
          P(idx(0, 0, 1), 5, p) = 6.961348543832062;
          P(idx(0, 0, 1), 6, p) = 3.5776874809726587;
          P(idx(0, 0, 1), 7, p) = -2.77507906596416;
          P(idx(0, 0, 1), 8, p) = 26.439340288997876;
          P(idx(0, 1, 0), 4, p) = -23.421601750764797;
          P(idx(0, 1, 0), 5, p) = 6.961348543832062;
          P(idx(0, 1, 0), 6, p) = 3.5776874809726587;
          P(idx(0, 1, 0), 7, p) = -2.77507906596416;
          P(idx(0, 1, 0), 8, p) = 5.172330290276515;
          P(idx(0, 1, 0), 9, p) = 29.398736610366683;
          P(idx(1, 0, 0), 5, p) = -19.491775922729772;
          P(idx(1, 0, 0), 6, p) = 4.79933686471942;
          P(idx(1, 0, 0), 7, p) = 5.368871559779907;
          P(idx(1, 0, 0), 8, p) = 4.527468332008274;
          P(idx(1, 0, 0), 9, p) = 29.398736610366683;
        }
      }
      else if (x0[p] + x1[p] > 0.5)
      {
        P(idx(0, 0), 5, p) = 19.491775922729772 - 19.491775922729772 * x2[p]
                             - 19.491775922729772 * x1[p]
                             - 19.491775922729772 * x0[p];
        P(idx(0, 0), 6, p) = -4.79933686471942 + 4.79933686471942 * x2[p]
                             + 4.79933686471942 * x1[p]
                             + 4.79933686471942 * x0[p];
        P(idx(0, 0), 7, p) = -17.793974312413408 + 30.219077065046907 * x2[p]
                             + 30.219077065046907 * x1[p]
                             + 5.368871559779907 * x0[p];
        P(idx(0, 0), 8, p) = 8.692201812490664 - 0.6448619582682409 * x2[p]
                             - 21.9118719569896 * x1[p]
                             + 4.527468332008274 * x0[p];
        P(idx(0, 0), 9, p) = -16.57019699857031 + 29.398736610366683 * x1[p]
                             + 29.398736610366683 * x0[p];
        if (nderiv >= 1)
        {
          P(idx(0, 0, 1), 5, p) = -19.491775922729772;
          P(idx(0, 0, 1), 6, p) = 4.79933686471942;
          P(idx(0, 0, 1), 7, p) = 30.219077065046907;
          P(idx(0, 0, 1), 8, p) = -0.6448619582682409;
          P(idx(0, 1, 0), 5, p) = -19.491775922729772;
          P(idx(0, 1, 0), 6, p) = 4.79933686471942;
          P(idx(0, 1, 0), 7, p) = 30.219077065046907;
          P(idx(0, 1, 0), 8, p) = -21.9118719569896;
          P(idx(0, 1, 0), 9, p) = 29.398736610366683;
          P(idx(1, 0, 0), 5, p) = -19.491775922729772;
          P(idx(1, 0, 0), 6, p) = 4.79933686471942;
          P(idx(1, 0, 0), 7, p) = 5.368871559779907;
          P(idx(1, 0, 0), 8, p) = 4.527468332008274;
          P(idx(1, 0, 0), 9, p) = 29.398736610366683;
        }
      }
      else
      {
        P(idx(0, 0), 5, p) = 9.745887961364886 - 19.491775922729772 * x2[p];
        P(idx(0, 0), 6, p) = 9.947716410509344 + 4.79933686471942 * x2[p]
                             - 24.69476968573811 * x1[p]
                             - 24.69476968573811 * x0[p];
        P(idx(0, 0), 7, p) = -17.110663386623237 + 30.219077065046907 * x2[p]
                             + 28.85245521346657 * x1[p]
                             + 4.002249708199567 * x0[p];
        P(idx(0, 0), 8, p) = -1.5584163991482487 - 0.6448619582682409 * x2[p]
                             - 1.4106355337117769 * x1[p]
                             + 25.0287047552861 * x0[p];
        P(idx(0, 0), 9, p) = 2.138089935299395 - 8.017837257372731 * x1[p]
                             - 8.017837257372731 * x0[p];
        if (nderiv >= 1)
        {
          P(idx(0, 0, 1), 5, p) = -19.491775922729772;
          P(idx(0, 0, 1), 6, p) = 4.79933686471942;
          P(idx(0, 0, 1), 7, p) = 30.219077065046907;
          P(idx(0, 0, 1), 8, p) = -0.6448619582682409;
          P(idx(0, 1, 0), 6, p) = -24.69476968573811;
          P(idx(0, 1, 0), 7, p) = 28.85245521346657;
          P(idx(0, 1, 0), 8, p) = -1.4106355337117769;
          P(idx(0, 1, 0), 9, p) = -8.017837257372731;
          P(idx(1, 0, 0), 6, p) = -24.69476968573811;
          P(idx(1, 0, 0), 7, p) = 4.002249708199567;
          P(idx(1, 0, 0), 8, p) = 25.0287047552861;
          P(idx(1, 0, 0), 9, p) = -8.017837257372731;
        }
      }
    }
  }
  else
  {
    throw std::runtime_error("Only degree 0 and 1 macro polysets are currently "
                             "implemented on a tetrahedron.");
  }
}
//-----------------------------------------------------------------------------
/// Compute the complete set of derivatives from 0 to nderiv, for all
/// the piecewise polynomials up to order n on a hexahedron split into 4 by
/// splitting each edge into two parts
template <typename T>
void tabulate_polyset_hexahedron_macroedge_derivs(
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 3>>
        P,
    std::size_t n, std::size_t nderiv,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        x)
{
  assert(x.extent(0) > 0);
  assert(P.extent(0) == (nderiv + 1) * (nderiv + 2) * (nderiv + 3) / 6);
  assert(P.extent(1) == (2 * n + 1) * (2 * n + 1) * (2 * n + 1));
  assert(P.extent(2) == x.extent(0));

  // Indexing for hexahedral basis functions
  auto hex_idx
      = [n](std::size_t px, std::size_t py, std::size_t pz) -> std::size_t
  { return (2 * n + 1) * (2 * n + 1) * px + (2 * n + 1) * py + pz; };

  auto x0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
  auto x1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 1);
  auto x2 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 2);

  std::fill(P.data_handle(), P.data_handle() + P.size(), 0.0);

  std::vector<T> factorials(n + 1, 0.0);

  // Fill with values of polynomials in x
  for (std::size_t k = 0; k <= n; ++k)
  {
    factorials[k] = (k % 2 == 0 ? 1 : -1) * single_choose(2 * n + 1 - k, n - k)
                    * single_choose(n, k) * pow(2, n - k);
  }
  for (std::size_t dx = 0; dx <= nderiv; ++dx)
  {
    for (std::size_t p = 0; p < P.extent(2); ++p)
    {
      T value = 0.0;
      if (x0[p] <= 0.5)
      {
        for (std::size_t k = 0; k + dx <= n; ++k)
        {
          T x_term = pow(x0[p], n - k - dx);
          for (std::size_t i = n - k; i > n - k - dx; --i)
            x_term *= i;
          value += factorials[k] * x_term;
        }
      }
      else
      {
        for (std::size_t k = 0; k + dx <= n; ++k)
        {
          T x_term = pow(1.0 - x0[p], n - k - dx);
          for (std::size_t i = n - k; i > n - k - dx; --i)
            x_term *= -i;
          value += factorials[k] * x_term;
        }
      }
      for (std::size_t dy = 0; dy <= nderiv - dx; ++dy)
        for (std::size_t dz = 0; dz <= nderiv - dy - dx; ++dz)
          for (std::size_t jy = 0; jy < 2 * n + 1; ++jy)
            for (std::size_t jz = 0; jz < 2 * n + 1; ++jz)
              P(idx(dx, dy, dz), hex_idx(0, jy, jz), p) = value;
    }
  }

  for (std::size_t j = 0; j < n; ++j)
  {
    for (std::size_t k = 0; k <= j; ++k)
    {
      factorials[k] = (k % 2 == 0 ? 1 : -1)
                      * single_choose(2 * n + 1 - k, j - k)
                      * single_choose(j, k) * pow(2, j - k) * pow(2, n - j)
                      * sqrt(4 * (n - j) + 2);
    }
    for (std::size_t dx = 0; dx <= nderiv; ++dx)
    {
      for (std::size_t p = 0; p < P.extent(2); ++p)
      {
        if (x0[p] <= 0.5)
        {
          T value = 0.0;
          for (std::size_t k = 0; k + dx <= j; ++k)
          {
            T x_term = pow(x0[p], j - k - dx);
            for (std::size_t i = j - k; i > j - k - dx; --i)
              x_term *= i;
            value += factorials[k] * x_term;
          }
          value *= pow(0.5 - x0[p], n - j - dx);
          for (std::size_t i = n - j; i > n - j - dx; --i)
            value *= -i;
          for (std::size_t dy = 0; dy <= nderiv - dx; ++dy)
            for (std::size_t dz = 0; dz <= nderiv - dy - dx; ++dz)
              for (std::size_t jy = 0; jy < 2 * n + 1; ++jy)
                for (std::size_t jz = 0; jz < 2 * n + 1; ++jz)
                  P(idx(dx, dy, dz), hex_idx(j + 1, jy, jz), p) = value;
        }
        else
        {
          T value = 0.0;
          for (std::size_t k = 0; k + dx <= j; ++k)
          {
            T x_term = pow(1.0 - x0[p], j - k - dx);
            for (std::size_t i = j - k; i > j - k - dx; --i)
              x_term *= -i;
            value += factorials[k] * x_term;
          }
          value *= pow(x0[p] - 0.5, n - j - dx);
          for (std::size_t i = n - j; i > n - j - dx; --i)
            value *= i;
          for (std::size_t dy = 0; dy <= nderiv - dx; ++dy)
            for (std::size_t dz = 0; dz <= nderiv - dy - dx; ++dz)
              for (std::size_t jy = 0; jy < 2 * n + 1; ++jy)
                for (std::size_t jz = 0; jz < 2 * n + 1; ++jz)
                  P(idx(dx, dy, dz), hex_idx(j + n + 1, jy, jz), p) = value;
        }
      }
    }
  }

  // Multiply by values of polynomials in y
  for (std::size_t k = 0; k <= n; ++k)
  {
    factorials[k] = (k % 2 == 0 ? 1 : -1) * single_choose(2 * n + 1 - k, n - k)
                    * single_choose(n, k) * pow(2, n - k);
  }
  for (std::size_t dy = 0; dy <= nderiv; ++dy)
  {
    for (std::size_t p = 0; p < P.extent(2); ++p)
    {
      T value = 0.0;
      if (x1[p] <= 0.5)
      {
        for (std::size_t k = 0; k + dy <= n; ++k)
        {
          T y_term = pow(x1[p], n - k - dy);
          for (std::size_t i = n - k; i > n - k - dy; --i)
            y_term *= i;
          value += factorials[k] * y_term;
        }
      }
      else
      {
        for (std::size_t k = 0; k + dy <= n; ++k)
        {
          T y_term = pow(1.0 - x1[p], n - k - dy);
          for (std::size_t i = n - k; i > n - k - dy; --i)
            y_term *= -i;
          value += factorials[k] * y_term;
        }
      }
      for (std::size_t dx = 0; dx <= nderiv - dy; ++dx)
        for (std::size_t dz = 0; dz <= nderiv - dx - dy; ++dz)
          for (std::size_t jx = 0; jx < 2 * n + 1; ++jx)
            for (std::size_t jz = 0; jz < 2 * n + 1; ++jz)
              P(idx(dx, dy, dz), hex_idx(jx, 0, jz), p) *= value;
    }
  }

  for (std::size_t j = 0; j < n; ++j)
  {
    for (std::size_t k = 0; k <= j; ++k)
    {
      factorials[k] = (k % 2 == 0 ? 1 : -1)
                      * single_choose(2 * n + 1 - k, j - k)
                      * single_choose(j, k) * pow(2, j - k) * pow(2, n - j)
                      * sqrt(4 * (n - j) + 2);
    }
    for (std::size_t dy = 0; dy <= nderiv; ++dy)
    {
      for (std::size_t p = 0; p < P.extent(2); ++p)
      {
        if (x1[p] <= 0.5)
        {
          T value = 0.0;
          for (std::size_t k = 0; k + dy <= j; ++k)
          {
            T y_term = pow(x1[p], j - k - dy);
            for (std::size_t i = j - k; i > j - k - dy; --i)
              y_term *= i;
            value += factorials[k] * y_term;
          }
          value *= pow(0.5 - x1[p], n - j - dy);
          for (std::size_t i = n - j; i > n - j - dy; --i)
            value *= -i;
          for (std::size_t dx = 0; dx <= nderiv - dy; ++dx)
            for (std::size_t dz = 0; dz <= nderiv - dx - dy; ++dz)
              for (std::size_t jx = 0; jx < 2 * n + 1; ++jx)
                for (std::size_t jz = 0; jz < 2 * n + 1; ++jz)
                {
                  P(idx(dx, dy, dz), hex_idx(jx, j + 1, jz), p) *= value;
                  P(idx(dx, dy, dz), hex_idx(jx, j + n + 1, jz), p) *= 0.0;
                }
        }
        else
        {
          T value = 0.0;
          for (std::size_t k = 0; k + dy <= j; ++k)
          {
            T y_term = pow(1.0 - x1[p], j - k - dy);
            for (std::size_t i = j - k; i > j - k - dy; --i)
              y_term *= -i;
            value += factorials[k] * y_term;
          }
          value *= pow(x1[p] - 0.5, n - j - dy);
          for (std::size_t i = n - j; i > n - j - dy; --i)
            value *= i;
          for (std::size_t dx = 0; dx <= nderiv - dy; ++dx)
          {
            for (std::size_t dz = 0; dz <= nderiv - dx - dy; ++dz)
            {
              for (std::size_t jx = 0; jx < 2 * n + 1; ++jx)
              {
                for (std::size_t jz = 0; jz < 2 * n + 1; ++jz)
                {
                  P(idx(dx, dy, dz), hex_idx(jx, j + 1, jz), p) *= 0.0;
                  P(idx(dx, dy, dz), hex_idx(jx, j + n + 1, jz), p) *= value;
                }
              }
            }
          }
        }
      }
    }
  }

  // Multiply by values of polynomials in z
  for (std::size_t k = 0; k <= n; ++k)
  {
    factorials[k] = (k % 2 == 0 ? 1 : -1) * single_choose(2 * n + 1 - k, n - k)
                    * single_choose(n, k) * pow(2, n - k);
  }
  for (std::size_t dz = 0; dz <= nderiv; ++dz)
  {
    for (std::size_t p = 0; p < P.extent(2); ++p)
    {
      T value = 0.0;
      if (x2[p] <= 0.5)
      {
        for (std::size_t k = 0; k + dz <= n; ++k)
        {
          T z_term = pow(x2[p], n - k - dz);
          for (std::size_t i = n - k; i > n - k - dz; --i)
            z_term *= i;
          value += factorials[k] * z_term;
        }
      }
      else
      {
        for (std::size_t k = 0; k + dz <= n; ++k)
        {
          T z_term = pow(1.0 - x2[p], n - k - dz);
          for (std::size_t i = n - k; i > n - k - dz; --i)
            z_term *= -i;
          value += factorials[k] * z_term;
        }
      }
      for (std::size_t dx = 0; dx <= nderiv - dz; ++dx)
        for (std::size_t dy = 0; dy <= nderiv - dx - dz; ++dy)
          for (std::size_t jx = 0; jx < 2 * n + 1; ++jx)
            for (std::size_t jy = 0; jy < 2 * n + 1; ++jy)
              P(idx(dx, dy, dz), hex_idx(jx, jy, 0), p) *= value;
    }
  }

  for (std::size_t j = 0; j < n; ++j)
  {
    for (std::size_t k = 0; k <= j; ++k)
    {
      factorials[k] = (k % 2 == 0 ? 1 : -1)
                      * single_choose(2 * n + 1 - k, j - k)
                      * single_choose(j, k) * pow(2, j - k) * pow(2, n - j)
                      * sqrt(4 * (n - j) + 2);
    }
    for (std::size_t dz = 0; dz <= nderiv; ++dz)
    {
      for (std::size_t p = 0; p < P.extent(2); ++p)
      {
        if (x2[p] <= 0.5)
        {
          T value = 0.0;
          for (std::size_t k = 0; k + dz <= j; ++k)
          {
            T z_term = pow(x2[p], j - k - dz);
            for (std::size_t i = j - k; i > j - k - dz; --i)
              z_term *= i;
            value += factorials[k] * z_term;
          }
          value *= pow(0.5 - x2[p], n - j - dz);
          for (std::size_t i = n - j; i > n - j - dz; --i)
            value *= -i;
          for (std::size_t dx = 0; dx <= nderiv - dz; ++dx)
            for (std::size_t dy = 0; dy <= nderiv - dx - dz; ++dy)
              for (std::size_t jx = 0; jx < 2 * n + 1; ++jx)
                for (std::size_t jy = 0; jy < 2 * n + 1; ++jy)
                {
                  P(idx(dx, dy, dz), hex_idx(jx, jy, j + 1), p) *= value;
                  P(idx(dx, dy, dz), hex_idx(jx, jy, j + n + 1), p) *= 0.0;
                }
        }
        else
        {
          T value = 0.0;
          for (std::size_t k = 0; k + dz <= j; ++k)
          {
            T z_term = pow(1.0 - x2[p], j - k - dz);
            for (std::size_t i = j - k; i > j - k - dz; --i)
              z_term *= -i;
            value += factorials[k] * z_term;
          }
          value *= pow(x2[p] - 0.5, n - j - dz);
          for (std::size_t i = n - j; i > n - j - dz; --i)
            value *= i;
          for (std::size_t dx = 0; dx <= nderiv - dz; ++dx)
          {
            for (std::size_t dy = 0; dy <= nderiv - dx - dz; ++dy)
            {
              for (std::size_t jx = 0; jx < 2 * n + 1; ++jx)
              {
                for (std::size_t jy = 0; jy < 2 * n + 1; ++jy)
                {
                  P(idx(dx, dy, dz), hex_idx(jx, jy, j + 1), p) *= 0.0;
                  P(idx(dx, dy, dz), hex_idx(jx, jy, j + n + 1), p) *= value;
                }
              }
            }
          }
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------

/// Compute the complete set of derivatives from 0 to nderiv, for all
/// the polynomials up to order n on a triangle in [0, 1][0, 1]. The
/// polynomials P_{pq} are built up in sequence, firstly along q = 0,
/// which is a line segment, as in tabulate_polyset_interval_derivs
/// above, but with a change of variables. The polynomials are then
/// extended in the q direction, using the relation given in Sherwin and
/// Karniadakis 1995 (https://doi.org/10.1016/0045-7825(94)00745-9).
template <typename T>
void tabulate_polyset_triangle_derivs(
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 3>>
        P,
    std::size_t n, std::size_t nderiv,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        x)
{
  assert(x.extent(1) == 2);

  auto x0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
  auto x1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 1);

  assert(P.extent(0) == (nderiv + 1) * (nderiv + 2) / 2);
  assert(P.extent(1) == (n + 1) * (n + 2) / 2);
  assert(P.extent(2) == x.extent(0));

  std::fill(P.data_handle(), P.data_handle() + P.size(), 0.0);
  if (n == 0)
  {
    for (std::size_t j = 0; j < P.extent(2); ++j)
      P(idx(0, 0), 0, j) = std::sqrt(2.0);
    return;
  }

  for (std::size_t j = 0; j < P.extent(2); ++j)
    P(idx(0, 0), 0, j) = 1.0;

  // Iterate over derivatives in increasing order, since higher
  // derivatives
  for (std::size_t kx = 0; kx <= nderiv; ++kx)
  {
    for (std::size_t ky = 0; ky <= nderiv - kx; ++ky)
    {
      for (std::size_t p = 1; p <= n; ++p)
      {
        auto p0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            P, idx(kx, ky), idx(0, p),
            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        auto p1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            P, idx(kx, ky), idx(0, p - 1),
            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        T a = static_cast<T>(2 * p - 1) / static_cast<T>(p);
        for (std::size_t i = 0; i < p0.extent(0); ++i)
        {
          p0[i] = ((x0[i] * 2.0 - 1.0) + 0.5 * (x1[i] * 2.0 - 1.0) + 0.5)
                  * p1[i] * a;
        }

        if (kx > 0)
        {
          auto px = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              P, idx(kx - 1, ky), idx(0, p - 1),
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
          for (std::size_t i = 0; i < p0.extent(0); ++i)
            p0[i] += 2 * kx * a * px[i];
        }

        if (ky > 0)
        {
          auto py = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              P, idx(kx, ky - 1), idx(0, p - 1),
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
          for (std::size_t i = 0; i < p0.extent(0); ++i)
            p0[i] += ky * a * py[i];
        }

        if (p > 1)
        {
          auto p2 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              P, idx(kx, ky), idx(0, p - 2),
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);

          // y^2 terms
          for (std::size_t i = 0; i < p0.extent(0); ++i)
          {
            const T f3 = 1.0 - x1[i];
            p0[i] -= f3 * f3 * p2[i] * (a - 1.0);
          }

          if (ky > 0)
          {
            auto p2y = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx, ky - 1), idx(0, p - 2),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            for (std::size_t i = 0; i < p0.extent(0); ++i)
              p0[i] -= ky * ((x1[i] * 2.0 - 1.0) - 1.0) * p2y[i] * (a - 1.0);
          }

          if (ky > 1)
          {
            auto p2y2 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx, ky - 2), idx(0, p - 2),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            for (std::size_t i = 0; i < p0.extent(0); ++i)
              p0[i] -= ky * (ky - 1) * p2y2[i] * (a - 1.0);
          }
        }
      }

      for (std::size_t p = 0; p < n; ++p)
      {
        auto p0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            P, idx(kx, ky), idx(0, p),
            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        auto p1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            P, idx(kx, ky), idx(1, p),
            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        for (std::size_t i = 0; i < p1.extent(0); ++i)
          p1[i] = p0[i] * ((x1[i] * 2.0 - 1.0) * (1.5 + p) + 0.5 + p);

        if (ky > 0)
        {
          auto py = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              P, idx(kx, ky - 1), idx(0, p),
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
          for (std::size_t i = 0; i < p1.size(); ++i)
            p1[i] += 2 * ky * (1.5 + p) * py[i];
        }

        for (std::size_t q = 1; q < n - p; ++q)
        {
          const auto [a1, a2, a3] = jrc<T>(2 * p + 1, q);
          auto pqp1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              P, idx(kx, ky), idx(q + 1, p),
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
          auto pqm1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              P, idx(kx, ky), idx(q - 1, p),
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
          auto pq = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              P, idx(kx, ky), idx(q, p),
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);

          for (std::size_t i = 0; i < pqp1.extent(0); ++i)
            pqp1[i] = pq[i] * ((x1[i] * 2.0 - 1.0) * a1 + a2) - pqm1[i] * a3;
          if (ky > 0)
          {
            auto py = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx, ky - 1), idx(q, p),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            for (std::size_t i = 0; i < pqp1.extent(0); ++i)
              pqp1[i] += 2 * ky * a1 * py[i];
          }
        }
      }
    }
  }

  // Normalisation
  {
    for (std::size_t i = 0; i < P.extent(0); ++i)
    {
      for (std::size_t p = 0; p <= n; ++p)
      {
        for (std::size_t q = 0; q <= n - p; ++q)
        {
          const T norm = std::sqrt((p + 0.5) * (p + q + 1)) * 2;
          const int j = idx(q, p);
          for (std::size_t k = 0; k < P.extent(2); ++k)
            P(i, j, k) *= norm;
        }
      }
    }
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void tabulate_polyset_tetrahedron_derivs(
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 3>>
        P,
    std::size_t n, std::size_t nderiv,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        x)
{
  assert(x.extent(1) == 3);
  assert(P.extent(0) == (nderiv + 1) * (nderiv + 2) * (nderiv + 3) / 6);
  assert(P.extent(1) == (n + 1) * (n + 2) * (n + 3) / 6);
  assert(P.extent(2) == x.extent(0));

  auto x0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
  auto x1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 1);
  auto x2 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 2);

  // Traverse derivatives in increasing order
  std::fill(P.data_handle(), P.data_handle() + P.size(), 0.0);
  for (std::size_t i = 0; i < P.extent(2); ++i)
    P(idx(0, 0, 0), 0, i) = 1.0;

  if (n == 0)
  {
    for (std::size_t i = 0; i < P.extent(2); ++i)
      P(idx(0, 0, 0), 0, i) = std::sqrt(6);
    return;
  }

  for (std::size_t kx = 0; kx <= nderiv; ++kx)
  {
    for (std::size_t ky = 0; ky <= nderiv - kx; ++ky)
    {
      for (std::size_t kz = 0; kz <= nderiv - kx - ky; ++kz)
      {
        for (std::size_t p = 1; p <= n; ++p)
        {
          auto p00 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              P, idx(kx, ky, kz), idx(0, 0, p),
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
          auto p0m1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              P, idx(kx, ky, kz), idx(0, 0, p - 1),
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
          T a = static_cast<T>(2 * p - 1) / static_cast<T>(p);
          for (std::size_t i = 0; i < p00.size(); ++i)
          {
            p00[i] = ((x0[i] * 2.0 - 1.0)
                      + 0.5 * ((x1[i] * 2.0 - 1.0) + (x2[i] * 2.0 - 1.0)) + 1.0)
                     * a * p0m1[i];
          }

          if (kx > 0)
          {
            auto p0m1x = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx - 1, ky, kz), idx(0, 0, p - 1),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            for (std::size_t i = 0; i < p00.size(); ++i)
              p00[i] += 2 * kx * a * p0m1x[i];
          }

          if (ky > 0)
          {
            auto p0m1y = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx, ky - 1, kz), idx(0, 0, p - 1),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            for (std::size_t i = 0; i < p00.size(); ++i)
              p00[i] += ky * a * p0m1y[i];
          }

          if (kz > 0)
          {
            auto p0m1z = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx, ky, kz - 1), idx(0, 0, p - 1),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            for (std::size_t i = 0; i < p00.size(); ++i)
              p00[i] += kz * a * p0m1z[i];
          }

          if (p > 1)
          {
            auto p0m2 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx, ky, kz), idx(0, 0, p - 2),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            for (std::size_t i = 0; i < p00.size(); ++i)
            {
              T f2 = x1[i] + x2[i] - 1.0;
              p00[i] -= f2 * f2 * p0m2[i] * (a - 1.0);
            }
            if (ky > 0)
            {
              auto p0m2y = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky - 1, kz), idx(0, 0, p - 2),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
              for (std::size_t i = 0; i < p00.size(); ++i)
              {
                p00[i] -= ky * ((x1[i] * 2.0 - 1.0) + (x2[i] * 2.0 - 1.0))
                          * p0m2y[i] * (a - 1.0);
              }
            }

            if (ky > 1)
            {
              auto p0m2y2 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky - 2, kz), idx(0, 0, p - 2),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
              for (std::size_t i = 0; i < p00.size(); ++i)
                p00[i] -= ky * (ky - 1) * p0m2y2[i] * (a - 1.0);
            }

            if (kz > 0)
            {
              auto p0m2z = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky, kz - 1), idx(0, 0, p - 2),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
              for (std::size_t i = 0; i < p00.size(); ++i)
                p00[i] -= kz * ((x1[i] * 2.0 - 1.0) + (x2[i] * 2.0 - 1.0))
                          * p0m2z[i] * (a - 1.0);
            }

            if (kz > 1)
            {
              auto p0m2z2 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky, kz - 2), idx(0, 0, p - 2),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
              for (std::size_t i = 0; i < p00.size(); ++i)
                p00[i] -= kz * (kz - 1) * p0m2z2[i] * (a - 1.0);
            }

            if (ky > 0 and kz > 0)
            {
              auto p0m2yz = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky - 1, kz - 1), idx(0, 0, p - 2),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
              for (std::size_t i = 0; i < p00.size(); ++i)
                p00[i] -= 2.0 * ky * kz * p0m2yz[i] * (a - 1.0);
            }
          }
        }

        for (std::size_t p = 0; p < n; ++p)
        {
          auto p10 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              P, idx(kx, ky, kz), idx(0, 1, p),
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
          auto p00 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              P, idx(kx, ky, kz), idx(0, 0, p),
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
          for (std::size_t i = 0; i < p10.size(); ++i)
            p10[i]
                = p00[i]
                  * ((1.0 + (x1[i] * 2.0 - 1.0)) * p
                     + (2.0 + (x1[i] * 2.0 - 1.0) * 3.0 + (x2[i] * 2.0 - 1.0))
                           * 0.5);
          if (ky > 0)
          {
            auto p0y = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx, ky - 1, kz), idx(0, 0, p),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            for (std::size_t i = 0; i < p10.size(); ++i)
              p10[i] += 2 * ky * p0y[i] * (1.5 + p);
          }

          if (kz > 0)
          {
            auto p0z = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx, ky, kz - 1), idx(0, 0, p),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            for (std::size_t i = 0; i < p10.size(); ++i)
              p10[i] += kz * p0z[i];
          }

          for (std::size_t q = 1; q < n - p; ++q)
          {
            auto [aq, bq, cq] = jrc<T>(2 * p + 1, q);
            auto pq1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx, ky, kz), idx(0, q + 1, p),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            auto pq = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx, ky, kz), idx(0, q, p),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            auto pqm1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx, ky, kz), idx(0, q - 1, p),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            for (std::size_t i = 0; i < pq1.size(); ++i)
            {
              T f4 = 1.0 - x2[i];
              T f3 = (x1[i] * 2.0 - 1.0 + x2[i]);
              pq1[i] = pq[i] * (f3 * aq + f4 * bq) - pqm1[i] * f4 * f4 * cq;
            }
            if (ky > 0)
            {
              auto pqy = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky - 1, kz), idx(0, q, p),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
              for (std::size_t i = 0; i < pq1.size(); ++i)
                pq1[i] += 2 * ky * pqy[i] * aq;
            }

            if (kz > 0)
            {
              auto pqz = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky, kz - 1), idx(0, q, p),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
              auto pq1z = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky, kz - 1), idx(0, q - 1, p),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
              for (std::size_t i = 0; i < pq1.size(); ++i)
              {
                pq1[i] += kz * pqz[i] * (aq - bq)
                          + kz * (1.0 - (x2[i] * 2.0 - 1.0)) * pq1z[i] * cq;
              }
            }

            if (kz > 1)
            {
              auto pq1z2 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky, kz - 2), idx(0, q - 1, p),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
              // Quadratic term in z
              for (std::size_t i = 0; i < pq1.size(); ++i)
                pq1[i] -= kz * (kz - 1) * pq1z2[i] * cq;
            }
          }
        }

        for (std::size_t p = 0; p < n; ++p)
        {
          for (std::size_t q = 0; q < n - p; ++q)
          {
            auto pq = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx, ky, kz), idx(1, q, p),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            auto pq0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx, ky, kz), idx(0, q, p),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            for (std::size_t i = 0; i < pq.size(); ++i)
            {
              pq[i] = pq0[i]
                      * ((1.0 + p + q) + (x2[i] * 2.0 - 1.0) * (2.0 + p + q));
            }

            if (kz > 0)
            {
              auto pqz = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky, kz - 1), idx(0, q, p),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
              for (std::size_t i = 0; i < pq.size(); ++i)
                pq[i] += 2 * kz * (2.0 + p + q) * pqz[i];
            }
          }
        }

        for (std::size_t p = 0; p + 1 < n; ++p)
        {
          for (std::size_t q = 0; q + 1 < n - p; ++q)
          {
            for (std::size_t r = 1; r < n - p - q; ++r)
            {
              auto [ar, br, cr] = jrc<T>(2 * p + 2 * q + 2, r);
              auto pqr1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky, kz), idx(r + 1, q, p),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
              auto pqr = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky, kz), idx(r, q, p),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
              auto pqrm1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky, kz), idx(r - 1, q, p),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);

              for (std::size_t i = 0; i < pqr1.size(); ++i)
              {
                pqr1[i]
                    = pqr[i] * ((x2[i] * 2.0 - 1.0) * ar + br) - pqrm1[i] * cr;
              }

              if (kz > 0)
              {
                auto pqrz = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                    P, idx(kx, ky, kz - 1), idx(r, q, p),
                    MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
                for (std::size_t i = 0; i < pqr1.size(); ++i)
                  pqr1[i] += 2 * kz * ar * pqrz[i];
              }
            }
          }
        }
      }
    }
  }

  // Normalise
  for (std::size_t p = 0; p <= n; ++p)
  {
    for (std::size_t q = 0; q <= n - p; ++q)
    {
      for (std::size_t r = 0; r <= n - p - q; ++r)
      {
        auto pqr = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            P, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, idx(r, q, p),
            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        for (std::size_t i = 0; i < pqr.extent(0); ++i)
          for (std::size_t j = 0; j < pqr.extent(1); ++j)
            pqr(i, j)
                *= std::sqrt(2 * (p + 0.5) * (p + q + 1.0) * (p + q + r + 1.5))
                   * 2;
      }
    }
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void tabulate_polyset_pyramid_derivs(
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 3>>
        P,
    std::size_t n, std::size_t nderiv,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        x)
{
  assert(x.extent(1) == 3);
  assert(P.extent(0) == (nderiv + 1) * (nderiv + 2) * (nderiv + 3) / 6);
  assert(P.extent(1) == (n + 1) * (n + 2) * (2 * n + 3) / 6);
  assert(P.extent(2) == x.extent(0));

  // Indexing for pyramidal basis functions
  auto pyr_idx = [n](std::size_t p, std::size_t q, std::size_t r) -> std::size_t
  {
    std::size_t rv = n - r + 1;
    std::size_t r0 = r * (n + 1) * (n - r + 2) + (2 * r - 1) * (r - 1) * r / 6;
    return r0 + p * rv + q;
  };

  const auto x0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
  const auto x1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 1);
  const auto x2 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 2);

  // Traverse derivatives in increasing order
  std::fill(P.data_handle(), P.data_handle() + P.size(), 0.0);

  for (std::size_t j = 0; j < P.extent(2); ++j)
    P(idx(0, 0, 0), pyr_idx(0, 0, 0), j) = 1.0;

  if (n == 0)
  {
    for (std::size_t j = 0; j < P.extent(2); ++j)
      P(idx(0, 0, 0), pyr_idx(0, 0, 0), j) = std::sqrt(3);
    return;
  }

  for (std::size_t k = 0; k <= nderiv; ++k)
  {
    for (std::size_t j = 0; j <= k; ++j)
    {
      for (std::size_t kx = 0; kx <= j; ++kx)
      {
        const std::size_t ky = j - kx;
        const std::size_t kz = k - j;

        // r = 0
        for (std::size_t p = 0; p <= n; ++p)
        {
          if (p > 0)
          {
            const T a = static_cast<T>(p - 1) / static_cast<T>(p);
            auto p00 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx, ky, kz), pyr_idx(p, 0, 0),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            auto p1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx, ky, kz), pyr_idx(p - 1, 0, 0),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            for (std::size_t i = 0; i < p00.size(); ++i)
              p00[i] = (0.5 + (x0[i] * 2.0 - 1.0) + (x2[i] * 2.0 - 1.0) * 0.5)
                       * p1[i] * (a + 1.0);

            if (kx > 0)
            {
              auto p11 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx - 1, ky, kz), pyr_idx(p - 1, 0, 0),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);

              for (std::size_t i = 0; i < p00.size(); ++i)
                p00[i] += 2.0 * kx * p11[i] * (a + 1.0);
            }

            if (kz > 0)
            {
              auto pz = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky, kz - 1), pyr_idx(p - 1, 0, 0),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);

              for (std::size_t i = 0; i < p00.size(); ++i)
                p00[i] += kz * pz[i] * (a + 1.0);
            }

            if (p > 1)
            {
              auto p2 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky, kz), pyr_idx(p - 2, 0, 0),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
              for (std::size_t i = 0; i < p00.size(); ++i)
              {
                T f2 = 1.0 - x2[i];
                p00[i] -= f2 * f2 * p2[i] * a;
              }

              if (kz > 0)
              {
                auto p2z = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                    P, idx(kx, ky, kz - 1), pyr_idx(p - 2, 0, 0),
                    MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
                for (std::size_t i = 0; i < p00.size(); ++i)
                  p00[i] += kz * (1.0 - (x2[i] * 2.0 - 1.0)) * p2z[i] * a;
              }

              if (kz > 1)
              {
                // quadratic term in z
                auto pz = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                    P, idx(kx, ky, kz - 2), pyr_idx(p - 2, 0, 0),
                    MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
                for (std::size_t i = 0; i < p00.size(); ++i)
                  p00[i] -= kz * (kz - 1) * pz[i] * a;
              }
            }
          }

          for (std::size_t q = 1; q < n + 1; ++q)
          {
            const T a = static_cast<T>(q - 1) / static_cast<T>(q);
            auto r_pq = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx, ky, kz), pyr_idx(p, q, 0),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);

            auto _p = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx, ky, kz), pyr_idx(p, q - 1, 0),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            for (std::size_t i = 0; i < r_pq.size(); ++i)
            {
              r_pq[i] = (0.5 + (x1[i] * 2.0 - 1.0) + (x2[i] * 2.0 - 1.0) * 0.5)
                        * _p[i] * (a + 1.0);
            }

            if (ky > 0)
            {
              auto _p = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky - 1, kz), pyr_idx(p, q - 1, 0),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
              for (std::size_t i = 0; i < r_pq.size(); ++i)
                r_pq[i] += 2.0 * ky * _p[i] * (a + 1.0);
            }

            if (kz > 0)
            {
              auto _p = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky, kz - 1), pyr_idx(p, q - 1, 0),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
              for (std::size_t i = 0; i < r_pq.size(); ++i)
                r_pq[i] += kz * _p[i] * (a + 1.0);
            }

            if (q > 1)
            {
              auto _p = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky, kz), pyr_idx(p, q - 2, 0),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
              for (std::size_t i = 0; i < r_pq.size(); ++i)
              {
                const T f2 = 1.0 - x2[i];
                r_pq[i] -= f2 * f2 * _p[i] * a;
              }

              if (kz > 0)
              {
                auto _p = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                    P, idx(kx, ky, kz - 1), pyr_idx(p, q - 2, 0),
                    MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
                for (std::size_t i = 0; i < r_pq.size(); ++i)
                  r_pq[i] += kz * (1.0 - (x2[i] * 2.0 - 1.0)) * _p[i] * a;
              }

              if (kz > 1)
              {
                auto _p = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                    P, idx(kx, ky, kz - 2), pyr_idx(p, q - 2, 0),
                    MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
                for (std::size_t i = 0; i < r_pq.size(); ++i)
                  r_pq[i] -= kz * (kz - 1) * _p[i] * a;
              }
            }
          }
        }

        // Extend into r > 0
        for (std::size_t p = 0; p < n; ++p)
        {
          for (std::size_t q = 0; q < n; ++q)
          {
            auto r_pq1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx, ky, kz), pyr_idx(p, q, 1),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);

            auto r_pq0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx, ky, kz), pyr_idx(p, q, 0),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            for (std::size_t i = 0; i < r_pq1.size(); ++i)
            {
              r_pq1[i]
                  = r_pq0[i]
                    * ((1.0 + p + q) + (x2[i] * 2.0 - 1.0) * (2.0 + p + q));
            }

            if (kz > 0)
            {
              auto r_pq = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky, kz - 1), pyr_idx(p, q, 0),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
              for (std::size_t i = 0; i < r_pq1.size(); ++i)
                r_pq1[i] += 2 * kz * r_pq[i] * (2.0 + p + q);
            }
          }
        }

        for (std::size_t r = 1; r <= n; ++r)
        {
          for (std::size_t p = 0; p < n - r; ++p)
          {
            for (std::size_t q = 0; q < n - r; ++q)
            {
              auto [ar, br, cr] = jrc<T>(2 * p + 2 * q + 2, r);
              auto r_pqr = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky, kz), pyr_idx(p, q, r + 1),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
              auto _r0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky, kz), pyr_idx(p, q, r),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
              auto _r1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                  P, idx(kx, ky, kz), pyr_idx(p, q, r - 1),
                  MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
              for (std::size_t i = 0; i < r_pqr.size(); ++i)
              {
                r_pqr[i]
                    = _r0[i] * ((x2[i] * 2.0 - 1.0) * ar + br) - _r1[i] * cr;
              }

              if (kz > 0)
              {
                auto _r = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                    P, idx(kx, ky, kz - 1), pyr_idx(p, q, r),
                    MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
                for (std::size_t i = 0; i < r_pqr.size(); ++i)
                  r_pqr[i] += ar * 2 * kz * _r[i];
              }
            }
          }
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
        auto pqr = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            P, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, pyr_idx(p, q, r),
            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        for (std::size_t i = 0; i < pqr.extent(0); ++i)
          for (std::size_t j = 0; j < pqr.extent(1); ++j)
            pqr(i, j)
                *= std::sqrt(2 * (q + 0.5) * (p + 0.5) * (p + q + r + 1.5)) * 2;
      }
    }
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void tabulate_polyset_quad_derivs(
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 3>>
        P,
    std::size_t n, std::size_t nderiv,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        x)
{
  assert(x.extent(1) == 2);
  assert(P.extent(0) == (nderiv + 1) * (nderiv + 2) / 2);
  assert(P.extent(1) == (n + 1) * (n + 1));
  assert(P.extent(2) == x.extent(0));

  // Indexing for quadrilateral basis functions
  auto quad_idx = [n](std::size_t px, std::size_t py) -> std::size_t
  { return (n + 1) * px + py; };

  // Compute 1D basis
  auto x0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
  auto x1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 1);

  assert(x0.extent(0) > 0);
  assert(x1.extent(0) > 0);

  // Compute tabulation of interval for px = 0
  std::fill(P.data_handle(), P.data_handle() + P.size(), 0.0);
  for (std::size_t j = 0; j < P.extent(2); ++j)
    P(idx(0, 0), quad_idx(0, 0), j) = 1.0;

  if (n == 0)
    return;

  { // scope
    auto result = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        P, idx(0, 0), MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < result.extent(1); ++i)
    {
      result(quad_idx(0, 1), i)
          = (x1[i] * 2.0 - 1.0) * result(quad_idx(0, 0), i);
    }

    for (std::size_t py = 2; py <= n; ++py)
    {
      const T a = 1.0 - 1.0 / static_cast<T>(py);
      for (std::size_t i = 0; i < result.extent(1); ++i)
      {
        result(quad_idx(0, py), i)
            = (x1[i] * 2.0 - 1.0) * result(quad_idx(0, py - 1), i) * (a + 1.0)
              - result(quad_idx(0, py - 2), i) * a;
      }
    }
  }
  for (std::size_t ky = 1; ky <= nderiv; ++ky)
  {
    // Get reference to this derivative
    auto result = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        P, idx(0, ky), MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    auto result0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        P, idx(0, ky - 1), MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t i = 0; i < result.extent(1); ++i)
    {
      result(quad_idx(0, 1), i)
          = (x1[i] * 2.0 - 1.0) * result(quad_idx(0, 0), i)
            + 2 * ky * result0(quad_idx(0, 0), i);
    }

    for (std::size_t py = 2; py <= n; ++py)
    {
      const T a = 1.0 - 1.0 / static_cast<T>(py);
      for (std::size_t i = 0; i < result.extent(1); ++i)
      {
        result(quad_idx(0, py), i)
            = (x1[i] * 2.0 - 1.0) * result(quad_idx(0, py - 1), i) * (a + 1.0)
              + 2 * ky * result0(quad_idx(0, py - 1), i) * (a + 1.0)
              - result(quad_idx(0, py - 2), i) * a;
      }
    }
  }

  // Take tensor product with another interval
  for (std::size_t ky = 0; ky <= nderiv; ++ky)
  {
    auto result = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        P, idx(0, ky), MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t py = 0; py <= n; ++py)
    {
      for (std::size_t i = 0; i < result.extent(1); ++i)
      {
        result(quad_idx(1, py), i)
            = (x0[i] * 2.0 - 1.0) * result(quad_idx(0, py), i);
      }
    }
  }

  for (std::size_t px = 2; px <= n; ++px)
  {
    const T a = 1.0 - 1.0 / static_cast<T>(px);
    for (std::size_t ky = 0; ky <= nderiv; ++ky)
    {
      auto result = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          P, idx(0, ky), MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      for (std::size_t py = 0; py <= n; ++py)
      {
        for (std::size_t i = 0; i < result.extent(1); ++i)
        {
          result(quad_idx(px, py), i) = (x0[i] * 2.0 - 1.0)
                                            * result(quad_idx(px - 1, py), i)
                                            * (a + 1.0)
                                        - result(quad_idx(px - 2, py), i) * a;
        }
      }
    }
  }

  for (std::size_t kx = 1; kx <= nderiv; ++kx)
  {
    for (std::size_t ky = 0; ky <= nderiv - kx; ++ky)
    {
      auto result = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          P, idx(kx, ky), MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      auto result0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          P, idx(kx - 1, ky), MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      for (std::size_t py = 0; py <= n; ++py)
      {
        for (std::size_t i = 0; i < result.extent(1); ++i)
        {
          result(quad_idx(1, py), i)
              = (x0[i] * 2.0 - 1.0) * result(quad_idx(0, py), i)
                + 2 * kx * result0(quad_idx(0, py), i);
        }
      }
    }

    for (std::size_t px = 2; px <= n; ++px)
    {
      const T a = 1.0 - 1.0 / static_cast<T>(px);
      for (std::size_t ky = 0; ky <= nderiv - kx; ++ky)
      {
        auto result = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            P, idx(kx, ky), MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        auto result0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            P, idx(kx - 1, ky), MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        for (std::size_t py = 0; py <= n; ++py)
        {
          for (std::size_t i = 0; i < result.extent(1); ++i)
          {
            result(quad_idx(px, py), i)
                = (x0[i] * 2.0 - 1.0) * result(quad_idx(px - 1, py), i)
                      * (a + 1.0)
                  + 2 * kx * result0(quad_idx(px - 1, py), i) * (a + 1.0)
                  - result(quad_idx(px - 2, py), i) * a;
          }
        }
      }
    }
  }

  // Normalise
  for (std::size_t px = 0; px <= n; ++px)
  {
    for (std::size_t py = 0; py <= n; ++py)
    {
      auto pxy = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          P, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, quad_idx(px, py),
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      for (std::size_t i = 0; i < pxy.extent(0); ++i)
        for (std::size_t j = 0; j < pxy.extent(1); ++j)
          pxy(i, j) *= std::sqrt((2 * px + 1) * (2 * py + 1));
    }
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void tabulate_polyset_hex_derivs(
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 3>>
        P,
    std::size_t n, std::size_t nderiv,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        x)
{
  assert(x.extent(1) == 3);
  assert(P.extent(0) == (nderiv + 1) * (nderiv + 2) * (nderiv + 3) / 6);
  assert(P.extent(1) == (n + 1) * (n + 1) * (n + 1));
  assert(P.extent(2) == x.extent(0));

  // Indexing for hexahedral basis functions
  auto hex_idx
      = [n](std::size_t px, std::size_t py, std::size_t pz) -> std::size_t
  { return (n + 1) * (n + 1) * px + (n + 1) * py + pz; };

  // Compute 1D basis
  auto x0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
  auto x1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 1);
  auto x2 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 2);
  assert(x0.extent(0) > 0);
  assert(x1.extent(0) > 0);
  assert(x2.extent(0) > 0);

  std::fill(P.data_handle(), P.data_handle() + P.size(), 0.0);
  for (std::size_t i = 0; i < P.extent(2); ++i)
    P(idx(0, 0, 0), hex_idx(0, 0, 0), i) = 1.0;

  if (n == 0)
    return;

  // Tabulate interval for px=py=0
  // For kz = 0
  { // scope
    auto result = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        P, idx(0, 0, 0), MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    // for pz = 1
    for (std::size_t i = 0; i < result.extent(1); ++i)
    {
      result(hex_idx(0, 0, 1), i)
          = (x2[i] * 2.0 - 1.0) * result(hex_idx(0, 0, 0), i);
    }

    // for larger values of pz
    for (std::size_t pz = 2; pz <= n; ++pz)
    {
      const T a = 1.0 - 1.0 / static_cast<T>(pz);
      for (std::size_t i = 0; i < result.extent(1); ++i)
      {
        result(hex_idx(0, 0, pz), i)
            = (x2[i] * 2.0 - 1.0) * result(hex_idx(0, 0, pz - 1), i) * (a + 1.0)
              - result(hex_idx(0, 0, pz - 2), i) * a;
      }
    }
  }

  // for larger values of kz
  for (std::size_t kz = 1; kz <= nderiv; ++kz)
  {
    // Get reference to this derivative
    auto result = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        P, idx(0, 0, kz), MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    auto result0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        P, idx(0, 0, kz - 1), MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    // for pz = 1
    for (std::size_t i = 0; i < result.extent(1); ++i)
    {
      result(hex_idx(0, 0, 1), i)
          = (x2[i] * 2.0 - 1.0) * result(hex_idx(0, 0, 0), i)
            + 2 * kz * result0(hex_idx(0, 0, 0), i);
    }

    // for larger values of pz
    for (std::size_t pz = 2; pz <= n; ++pz)
    {
      const T a = 1.0 - 1.0 / static_cast<T>(pz);
      for (std::size_t i = 0; i < result.extent(1); ++i)
      {
        result(hex_idx(0, 0, pz), i)
            = (x2[i] * 2.0 - 1.0) * result(hex_idx(0, 0, pz - 1), i) * (a + 1.0)
              + 2 * kz * result0(hex_idx(0, 0, pz - 1), i) * (a + 1.0)
              - result(hex_idx(0, 0, pz - 2), i) * a;
      }
    }
  }

  // Take tensor product with interval to get quad for px=0
  // for ky = 0
  // for py = 1
  for (std::size_t kz = 0; kz <= nderiv; ++kz)
  {
    auto result = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
        P, idx(0, 0, kz), MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
        MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
    for (std::size_t pz = 0; pz <= n; ++pz)
    {
      for (std::size_t i = 0; i < result.extent(1); ++i)
      {
        result(hex_idx(0, 1, pz), i)
            = (x1[i] * 2.0 - 1.0) * result(hex_idx(0, 0, pz), i);
      }
    }
  }

  for (std::size_t py = 2; py <= n; ++py)
  {
    const T a = 1.0 - 1.0 / static_cast<T>(py);
    for (std::size_t kz = 0; kz <= nderiv; ++kz)
    {
      auto result = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          P, idx(0, 0, kz), MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      for (std::size_t pz = 0; pz <= n; ++pz)
      {
        for (std::size_t i = 0; i < result.extent(1); ++i)
        {
          result(hex_idx(0, py, pz), i)
              = (x1[i] * 2.0 - 1.0) * result(hex_idx(0, py - 1, pz), i)
                    * (a + 1.0)
                - result(hex_idx(0, py - 2, pz), i) * a;
        }
      }
    }
  }

  // for larger values of ky
  for (std::size_t ky = 1; ky <= nderiv; ++ky)
  {
    // for py = 1
    for (std::size_t kz = 0; kz <= nderiv - ky; ++kz)
    {
      auto result = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          P, idx(0, ky, kz), MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      auto result0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          P, idx(0, ky - 1, kz), MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      for (std::size_t pz = 0; pz <= n; ++pz)
      {
        for (std::size_t i = 0; i < result.extent(1); ++i)
        {
          result(hex_idx(0, 1, pz), i)
              = (x1[i] * 2.0 - 1.0) * result(hex_idx(0, 0, pz), i)
                + 2 * ky * result0(hex_idx(0, 0, pz), i);
        }
      }
    }

    for (std::size_t py = 2; py <= n; ++py)
    {
      const T a = 1.0 - 1.0 / static_cast<T>(py);
      for (std::size_t kz = 0; kz <= nderiv - ky; ++kz)
      {
        auto result = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            P, idx(0, ky, kz), MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        auto result0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            P, idx(0, ky - 1, kz), MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        for (std::size_t pz = 0; pz <= n; ++pz)
        {
          for (std::size_t i = 0; i < result.extent(1); ++i)
          {
            result(hex_idx(0, py, pz), i)
                = (x1[i] * 2.0 - 1.0) * result(hex_idx(0, py - 1, pz), i)
                      * (a + 1.0)
                  + 2 * ky * result0(hex_idx(0, py - 1, pz), i) * (a + 1.0)
                  - result(hex_idx(0, py - 2, pz), i) * a;
          }
        }
      }
    }
  }

  // Take tensor product with interval to get hex
  // kx = 0
  // for px = 1
  for (std::size_t ky = 0; ky <= nderiv; ++ky)
  {
    for (std::size_t kz = 0; kz <= nderiv - ky; ++kz)
    {
      auto result = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          P, idx(0, ky, kz), MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
          MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      for (std::size_t py = 0; py <= n; ++py)
      {
        for (std::size_t pz = 0; pz <= n; ++pz)
        {
          for (std::size_t i = 0; i < result.extent(1); ++i)
          {
            result(hex_idx(1, py, pz), i)
                = (x0[i] * 2.0 - 1.0) * result(hex_idx(0, py, pz), i);
          }
        }
      }
    }
  }

  // For larger values of px
  for (std::size_t px = 2; px <= n; ++px)
  {
    const T a = 1.0 - 1.0 / static_cast<T>(px);
    for (std::size_t ky = 0; ky <= nderiv; ++ky)
    {
      for (std::size_t kz = 0; kz <= nderiv - ky; ++kz)
      {
        auto result = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            P, idx(0, ky, kz), MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        for (std::size_t py = 0; py <= n; ++py)
        {
          for (std::size_t pz = 0; pz <= n; ++pz)
          {
            for (std::size_t i = 0; i < result.extent(1); ++i)
            {
              result(hex_idx(px, py, pz), i)
                  = (x0[i] * 2.0 - 1.0) * result(hex_idx(px - 1, py, pz), i)
                        * (a + 1.0)
                    - result(hex_idx(px - 2, py, pz), i) * a;
            }
          }
        }
      }
    }
  }

  // For larger values of kx
  for (std::size_t kx = 1; kx <= nderiv; ++kx)
  {
    // for px = 1
    {
      for (std::size_t ky = 0; ky <= nderiv - kx; ++ky)
      {
        for (std::size_t kz = 0; kz <= nderiv - kx - ky; ++kz)
        {
          auto result = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              P, idx(kx, ky, kz), MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
          auto result0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              P, idx(kx - 1, ky, kz),
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
          for (std::size_t py = 0; py <= n; ++py)
          {
            for (std::size_t pz = 0; pz <= n; ++pz)
            {
              for (std::size_t i = 0; i < result.extent(1); ++i)
              {
                result(hex_idx(1, py, pz), i)
                    = (x0[i] * 2.0 - 1.0) * result(hex_idx(0, py, pz), i)
                      + 2 * kx * result0(hex_idx(0, py, pz), i);
              }
            }
          }
        }
      }
    }

    // For larger values of px
    for (std::size_t px = 2; px <= n; ++px)
    {
      const T a = 1.0 - 1.0 / static_cast<T>(px);
      for (std::size_t ky = 0; ky <= nderiv - kx; ++ky)
      {
        for (std::size_t kz = 0; kz <= nderiv - kx - ky; ++kz)
        {
          auto result = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              P, idx(kx, ky, kz), MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
          auto result0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              P, idx(kx - 1, ky, kz),
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
          for (std::size_t py = 0; py <= n; ++py)
          {
            for (std::size_t pz = 0; pz <= n; ++pz)
            {
              for (std::size_t i = 0; i < result.extent(1); ++i)
              {
                result(hex_idx(px, py, pz), i)
                    = (x0[i] * 2.0 - 1.0) * result(hex_idx(px - 1, py, pz), i)
                          * (a + 1.0)
                      + 2 * kx * result0(hex_idx(px - 1, py, pz), i) * (a + 1.0)
                      - result(hex_idx(px - 2, py, pz), i) * a;
              }
            }
          }
        }
      }
    }
  }

  // Normalise
  for (std::size_t px = 0; px <= n; ++px)
  {
    for (std::size_t py = 0; py <= n; ++py)
    {
      for (std::size_t pz = 0; pz <= n; ++pz)
      {
        auto pxyz = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            P, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, hex_idx(px, py, pz),
            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        for (std::size_t i = 0; i < pxyz.extent(0); ++i)
          for (std::size_t j = 0; j < pxyz.extent(1); ++j)
            pxyz(i, j) *= std::sqrt((2 * px + 1) * (2 * py + 1) * (2 * pz + 1));
      }
    }
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void tabulate_polyset_prism_derivs(
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 3>>
        P,
    std::size_t n, std::size_t nderiv,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        x)
{
  assert(x.extent(1) == 3);
  assert(P.extent(0) == (nderiv + 1) * (nderiv + 2) * (nderiv + 3) / 6);
  assert(P.extent(1) == (n + 1) * (n + 1) * (n + 2) / 2);
  assert(P.extent(2) == x.extent(0));

  auto x0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 0);
  auto x1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 1);
  auto x2 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
      x, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, 2);

  assert(x0.extent(0) > 0);
  assert(x1.extent(0) > 0);
  assert(x2.extent(0) > 0);

  // Indexing for hexahedral basis functions
  auto prism_idx
      = [n](std::size_t px, std::size_t py, std::size_t pz) -> std::size_t
  { return (n + 1) * idx(py, px) + pz; };

  // Tabulate triangle for px=0
  std::fill(P.data_handle(), P.data_handle() + P.size(), 0.0);
  if (n == 0)
  {
    for (std::size_t i = 0; i < P.extent(2); ++i)
      P(idx(0, 0, 0), prism_idx(0, 0, 0), i) = std::sqrt(2);
    return;
  }

  for (std::size_t i = 0; i < P.extent(2); ++i)
    P(idx(0, 0, 0), prism_idx(0, 0, 0), i) = 1.0;

  for (std::size_t kx = 0; kx <= nderiv; ++kx)
  {
    for (std::size_t ky = 0; ky <= nderiv - kx; ++ky)
    {
      for (std::size_t p = 1; p <= n; ++p)
      {
        auto p0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            P, idx(kx, ky, 0), prism_idx(p, 0, 0),
            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        auto p1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            P, idx(kx, ky, 0), prism_idx(p - 1, 0, 0),
            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        const T a = static_cast<T>(2 * p - 1) / static_cast<T>(p);
        for (std::size_t i = 0; i < p0.size(); ++i)
        {
          p0[i] = ((x0[i] * 2.0 - 1.0) + 0.5 * (x1[i] * 2.0 - 1.0) + 0.5)
                  * p1[i] * a;
        }

        if (kx > 0)
        {
          auto result0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              P, idx(kx - 1, ky, 0), prism_idx(p - 1, 0, 0),
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
          for (std::size_t i = 0; i < p0.size(); ++i)
            p0[i] += 2 * kx * a * result0[i];
        }

        if (ky > 0)
        {
          auto result0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              P, idx(kx, ky - 1, 0), prism_idx(p - 1, 0, 0),
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
          for (std::size_t i = 0; i < p0.size(); ++i)
            p0[i] += ky * a * result0[i];
        }

        if (p > 1)
        {
          // y^2 terms
          auto p2 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              P, idx(kx, ky, 0), prism_idx(p - 2, 0, 0),
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
          for (std::size_t i = 0; i < p0.size(); ++i)
          {
            T f2 = (1.0 - x1[i]);
            p0[i] -= f2 * f2 * p2[i] * (a - 1.0);
          }

          if (ky > 0)
          {
            auto result0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx, ky - 1, 0), prism_idx(p - 2, 0, 0),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            for (std::size_t i = 0; i < p0.size(); ++i)
            {
              p0[i]
                  -= ky * ((x1[i] * 2.0 - 1.0) - 1.0) * result0[i] * (a - 1.0);
            }
          }

          if (ky > 1)
          {
            auto result0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx, ky - 2, 0), prism_idx(p - 2, 0, 0),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            for (std::size_t i = 0; i < p0.size(); ++i)
              p0[i] -= ky * (ky - 1) * result0[i] * (a - 1.0);
          }
        }
      }

      for (std::size_t p = 0; p < n; ++p)
      {
        auto p0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            P, idx(kx, ky, 0), prism_idx(p, 0, 0),
            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        auto p1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            P, idx(kx, ky, 0), prism_idx(p, 1, 0),
            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        for (std::size_t i = 0; i < p1.size(); ++i)
          p1[i] = p0[i] * ((x1[i] * 2.0 - 1.0) * (1.5 + p) + 0.5 + p);

        if (ky > 0)
        {
          auto result0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              P, idx(kx, ky - 1, 0), prism_idx(p, 0, 0),
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
          for (std::size_t i = 0; i < p1.size(); ++i)
            p1[i] += 2 * ky * (1.5 + p) * result0[i];
        }

        for (std::size_t q = 1; q < n - p; ++q)
        {
          auto pqp1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              P, idx(kx, ky, 0), prism_idx(p, q + 1, 0),
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
          auto pq = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              P, idx(kx, ky, 0), prism_idx(p, q, 0),
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
          auto pqm1 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              P, idx(kx, ky, 0), prism_idx(p, q - 1, 0),
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
          const auto [a1, a2, a3] = jrc<T>(2 * p + 1, q);
          for (std::size_t i = 0; i < p0.size(); ++i)
            pqp1[i] = pq[i] * ((x1[i] * 2.0 - 1.0) * a1 + a2) - pqm1[i] * a3;

          if (ky > 0)
          {
            auto result0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
                P, idx(kx, ky - 1, 0), prism_idx(p, q, 0),
                MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
            for (std::size_t i = 0; i < pqp1.size(); ++i)
              pqp1[i] += 2 * ky * a1 * result0[i];
          }
        }
      }
    }
  }

  // Take tensor product with interval to get prism
  for (std::size_t kz = 0; kz <= nderiv; ++kz)
  {
    for (std::size_t r = 1; r <= n; ++r)
    {
      const T a = 1.0 - 1.0 / static_cast<T>(r);
      for (std::size_t kx = 0; kx <= nderiv - kz; ++kx)
      {
        for (std::size_t ky = 0; ky <= nderiv - kx - kz; ++ky)
        {
          auto result = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              P, idx(kx, ky, kz), MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
          auto result0 = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
              P, idx(kx, ky, kz - 1),
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent,
              MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
          for (std::size_t p = 0; p <= n; ++p)
          {
            for (std::size_t q = 0; q <= n - p; ++q)
            {
              for (std::size_t i = 0; i < result.extent(1); ++i)
              {
                result(prism_idx(p, q, r), i)
                    = (x2[i] * 2.0 - 1.0) * result(prism_idx(p, q, r - 1), i)
                      * (a + 1.0);
              }

              if (kz > 0)
              {
                for (std::size_t i = 0; i < result.extent(1); ++i)
                {
                  result(prism_idx(p, q, r), i)
                      += 2 * kz * result0(prism_idx(p, q, r - 1), i)
                         * (a + 1.0);
                }
              }

              if (r > 1)
              {
                for (std::size_t i = 0; i < result.extent(1); ++i)
                {
                  result(prism_idx(p, q, r), i)
                      -= result(prism_idx(p, q, r - 2), i) * a;
                }
              }
            }
          }
        }
      }
    }
  }

  // Normalise
  for (std::size_t p = 0; p <= n; ++p)
  {
    for (std::size_t q = 0; q <= n - p; ++q)
    {
      for (std::size_t r = 0; r <= n; ++r)
      {
        auto pqr = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
            P, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent, prism_idx(p, q, r),
            MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
        for (std::size_t i = 0; i < pqr.extent(0); ++i)
          for (std::size_t j = 0; j < pqr.extent(1); ++j)
            pqr(i, j) *= std::sqrt((p + 0.5) * (p + q + 1) * (2 * r + 1)) * 2;
      }
    }
  }
}
} // namespace
//-----------------------------------------------------------------------------
template <std::floating_point T>
void polyset::tabulate(
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 3>>
        P,
    cell::type celltype, polyset::type ptype, int d, int n,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        x)
{
  switch (ptype)
  {
  case polyset::type::standard:
    switch (celltype)
    {
    case cell::type::point:
      tabulate_polyset_point_derivs(P, d, n, x);
      return;
    case cell::type::interval:
      tabulate_polyset_line_derivs(P, d, n, x);
      return;
    case cell::type::triangle:
      tabulate_polyset_triangle_derivs(P, d, n, x);
      return;
    case cell::type::tetrahedron:
      tabulate_polyset_tetrahedron_derivs(P, d, n, x);
      return;
    case cell::type::quadrilateral:
      tabulate_polyset_quad_derivs(P, d, n, x);
      return;
    case cell::type::prism:
      tabulate_polyset_prism_derivs(P, d, n, x);
      return;
    case cell::type::pyramid:
      tabulate_polyset_pyramid_derivs(P, d, n, x);
      return;
    case cell::type::hexahedron:
      tabulate_polyset_hex_derivs(P, d, n, x);
      return;
    default:
      throw std::runtime_error("Polynomial set: unsupported cell type");
    }
  case polyset::type::macroedge:
    switch (celltype)
    {
    case cell::type::point:
      tabulate_polyset_point_derivs(P, d, n, x);
      return;
    case cell::type::interval:
      tabulate_polyset_line_macroedge_derivs(P, d, n, x);
      return;
    case cell::type::triangle:
      tabulate_polyset_triangle_macroedge_derivs(P, d, n, x);
      return;
    case cell::type::tetrahedron:
      tabulate_polyset_tetrahedron_macroedge_derivs(P, d, n, x);
      return;
    case cell::type::quadrilateral:
      tabulate_polyset_quadrilateral_macroedge_derivs(P, d, n, x);
      return;
    case cell::type::hexahedron:
      tabulate_polyset_hexahedron_macroedge_derivs(P, d, n, x);
      return;
    default:
      throw std::runtime_error("Polynomial set: unsupported cell type");
    }
  default:
    throw std::runtime_error("Polynomial set: unsupported polynomial type.");
  }
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 3>> polyset::tabulate(
    cell::type celltype, polyset::type ptype, int d, int n,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
        x)
{
  std::array<std::size_t, 3> shape
      = {(std::size_t)polyset::nderivs(celltype, n),
         (std::size_t)polyset::dim(celltype, ptype, d), x.extent(0)};
  std::vector<T> P(shape[0] * shape[1] * shape[2]);
  MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 3>>
      _P(P.data(), shape);
  polyset::tabulate(_P, celltype, ptype, d, n, x);
  return {std::move(P), std::move(shape)};
}
//-----------------------------------------------------------------------------
/// @cond
template std::pair<std::vector<float>, std::array<std::size_t, 3>>
polyset::tabulate(
    cell::type, polyset::type, int, int,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const float, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>);
template std::pair<std::vector<double>, std::array<std::size_t, 3>>
polyset::tabulate(
    cell::type, polyset::type, int, int,
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
        const double,
        MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>);
/// @endcond
//-----------------------------------------------------------------------------
int polyset::dim(cell::type celltype, polyset::type ptype, int d)
{
  switch (ptype)
  {
  case polyset::type::standard:
    switch (celltype)
    {
    case cell::type::point:
      return 1;
    case cell::type::triangle:
      return (d + 1) * (d + 2) / 2;
    case cell::type::tetrahedron:
      return (d + 1) * (d + 2) * (d + 3) / 6;
    case cell::type::prism:
      return (d + 1) * (d + 1) * (d + 2) / 2;
    case cell::type::pyramid:
      return (d + 1) * (d + 2) * (2 * d + 3) / 6;
    case cell::type::interval:
      return (d + 1);
    case cell::type::quadrilateral:
      return (d + 1) * (d + 1);
    case cell::type::hexahedron:
      return (d + 1) * (d + 1) * (d + 1);
    default:
      return 1;
    }
  case polyset::type::macroedge:
    switch (celltype)
    {
    case cell::type::point:
      return 1;
    case cell::type::interval:
      return 2 * d + 1;
    case cell::type::triangle:
      return (d + 1) * (2 * d + 1);
    case cell::type::tetrahedron:
      return (d + 1) * (2 * d + 1) * (2 * d + 3) / 3;
    case cell::type::quadrilateral:
      return (2 * d + 1) * (2 * d + 1);
    case cell::type::hexahedron:
      return (2 * d + 1) * (2 * d + 1) * (2 * d + 1);
    default:
      return 1;
    }
  default:
    return 1;
  }
}
//-----------------------------------------------------------------------------
int polyset::nderivs(cell::type celltype, int n)
{
  switch (celltype)
  {
  case cell::type::point:
    return 1;
  case cell::type::interval:
    return n + 1;
  case cell::type::triangle:
    return (n + 1) * (n + 2) / 2;
  case cell::type::quadrilateral:
    return (n + 1) * (n + 2) / 2;
  case cell::type::tetrahedron:
    return (n + 1) * (n + 2) * (n + 3) / 6;
  case cell::type::hexahedron:
    return (n + 1) * (n + 2) * (n + 3) / 6;
  case cell::type::prism:
    return (n + 1) * (n + 2) * (n + 3) / 6;
  case cell::type::pyramid:
    return (n + 1) * (n + 2) * (n + 3) / 6;
  default:
    return 1;
  }
}
//-----------------------------------------------------------------------------
polyset::type polyset::superset(cell::type, polyset::type type1,
                                polyset::type type2)
{
  if (type1 == type2)
    return type1;
  if (type1 == polyset::type::standard)
    return type2;
  if (type2 == polyset::type::standard)
    return type1;
  throw std::runtime_error("Unsupported superset of polynomial sets.");
}
//-----------------------------------------------------------------------------
polyset::type polyset::restriction(polyset::type ptype, cell::type cell,
                                   cell::type restriction_cell)
{
  if (ptype == polyset::type::standard)
    return polyset::type::standard;
  if (cell == restriction_cell)
    return ptype;
  throw std::runtime_error("Unsupported restriction of polynomial sets.");
}
//-----------------------------------------------------------------------------
