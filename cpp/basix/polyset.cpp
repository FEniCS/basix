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

namespace stdex = std::experimental;

namespace
{
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
    stdex::mdspan<T, stdex::dextents<std::size_t, 3>> P, std::size_t,
    std::size_t nderiv,
    stdex::mdspan<const T, stdex::dextents<std::size_t, 2>> x)
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
    stdex::mdspan<T, stdex::dextents<std::size_t, 3>> P, std::size_t n,
    std::size_t nderiv,
    stdex::mdspan<const T, stdex::dextents<std::size_t, 2>> x)
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

  auto x0 = stdex::submdspan(x, stdex::full_extent, 0);

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
/// the polynomials up to order n on a triangle in [0, 1][0, 1]. The
/// polynomials P_{pq} are built up in sequence, firstly along q = 0,
/// which is a line segment, as in tabulate_polyset_interval_derivs
/// above, but with a change of variables. The polynomials are then
/// extended in the q direction, using the relation given in Sherwin and
/// Karniadakis 1995 (https://doi.org/10.1016/0045-7825(94)00745-9).
template <typename T>
void tabulate_polyset_triangle_derivs(
    stdex::mdspan<T, stdex::dextents<std::size_t, 3>> P, std::size_t n,
    std::size_t nderiv,
    stdex::mdspan<const T, stdex::dextents<std::size_t, 2>> x)
{
  assert(x.extent(1) == 2);

  auto x0 = stdex::submdspan(x, stdex::full_extent, 0);
  auto x1 = stdex::submdspan(x, stdex::full_extent, 1);

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
        auto p0
            = stdex::submdspan(P, idx(kx, ky), idx(0, p), stdex::full_extent);
        auto p1 = stdex::submdspan(P, idx(kx, ky), idx(0, p - 1),
                                   stdex::full_extent);
        T a = static_cast<T>(2 * p - 1) / static_cast<T>(p);
        for (std::size_t i = 0; i < p0.extent(0); ++i)
        {
          p0[i] = ((x0[i] * 2.0 - 1.0) + 0.5 * (x1[i] * 2.0 - 1.0) + 0.5)
                  * p1[i] * a;
        }

        if (kx > 0)
        {
          auto px = stdex::submdspan(P, idx(kx - 1, ky), idx(0, p - 1),
                                     stdex::full_extent);
          for (std::size_t i = 0; i < p0.extent(0); ++i)
            p0[i] += 2 * kx * a * px[i];
        }

        if (ky > 0)
        {
          auto py = stdex::submdspan(P, idx(kx, ky - 1), idx(0, p - 1),
                                     stdex::full_extent);
          for (std::size_t i = 0; i < p0.extent(0); ++i)
            p0[i] += ky * a * py[i];
        }

        if (p > 1)
        {
          auto p2 = stdex::submdspan(P, idx(kx, ky), idx(0, p - 2),
                                     stdex::full_extent);

          // y^2 terms
          for (std::size_t i = 0; i < p0.extent(0); ++i)
          {
            const T f3 = 1.0 - x1[i];
            p0[i] -= f3 * f3 * p2[i] * (a - 1.0);
          }

          if (ky > 0)
          {
            auto p2y = stdex::submdspan(P, idx(kx, ky - 1), idx(0, p - 2),
                                        stdex::full_extent);
            for (std::size_t i = 0; i < p0.extent(0); ++i)
              p0[i] -= ky * ((x1[i] * 2.0 - 1.0) - 1.0) * p2y[i] * (a - 1.0);
          }

          if (ky > 1)
          {
            auto p2y2 = stdex::submdspan(P, idx(kx, ky - 2), idx(0, p - 2),
                                         stdex::full_extent);
            for (std::size_t i = 0; i < p0.extent(0); ++i)
              p0[i] -= ky * (ky - 1) * p2y2[i] * (a - 1.0);
          }
        }
      }

      for (std::size_t p = 0; p < n; ++p)
      {
        auto p0
            = stdex::submdspan(P, idx(kx, ky), idx(0, p), stdex::full_extent);
        auto p1
            = stdex::submdspan(P, idx(kx, ky), idx(1, p), stdex::full_extent);
        for (std::size_t i = 0; i < p1.extent(0); ++i)
          p1[i] = p0[i] * ((x1[i] * 2.0 - 1.0) * (1.5 + p) + 0.5 + p);

        if (ky > 0)
        {
          auto py = stdex::submdspan(P, idx(kx, ky - 1), idx(0, p),
                                     stdex::full_extent);
          for (std::size_t i = 0; i < p1.size(); ++i)
            p1[i] += 2 * ky * (1.5 + p) * py[i];
        }

        for (std::size_t q = 1; q < n - p; ++q)
        {
          const auto [a1, a2, a3] = jrc<T>(2 * p + 1, q);
          auto pqp1 = stdex::submdspan(P, idx(kx, ky), idx(q + 1, p),
                                       stdex::full_extent);
          auto pqm1 = stdex::submdspan(P, idx(kx, ky), idx(q - 1, p),
                                       stdex::full_extent);
          auto pq
              = stdex::submdspan(P, idx(kx, ky), idx(q, p), stdex::full_extent);

          for (std::size_t i = 0; i < pqp1.extent(0); ++i)
            pqp1[i] = pq[i] * ((x1[i] * 2.0 - 1.0) * a1 + a2) - pqm1[i] * a3;
          if (ky > 0)
          {
            auto py = stdex::submdspan(P, idx(kx, ky - 1), idx(q, p),
                                       stdex::full_extent);
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
    stdex::mdspan<T, stdex::dextents<std::size_t, 3>> P, std::size_t n,
    std::size_t nderiv,
    stdex::mdspan<const T, stdex::dextents<std::size_t, 2>> x)
{
  assert(x.extent(1) == 3);
  assert(P.extent(0) == (nderiv + 1) * (nderiv + 2) * (nderiv + 3) / 6);
  assert(P.extent(1) == (n + 1) * (n + 2) * (n + 3) / 6);
  assert(P.extent(2) == x.extent(0));

  auto x0 = stdex::submdspan(x, stdex::full_extent, 0);
  auto x1 = stdex::submdspan(x, stdex::full_extent, 1);
  auto x2 = stdex::submdspan(x, stdex::full_extent, 2);

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
          auto p00 = stdex::submdspan(P, idx(kx, ky, kz), idx(0, 0, p),
                                      stdex::full_extent);
          auto p0m1 = stdex::submdspan(P, idx(kx, ky, kz), idx(0, 0, p - 1),
                                       stdex::full_extent);
          T a = static_cast<T>(2 * p - 1) / static_cast<T>(p);
          for (std::size_t i = 0; i < p00.size(); ++i)
          {
            p00[i] = ((x0[i] * 2.0 - 1.0)
                      + 0.5 * ((x1[i] * 2.0 - 1.0) + (x2[i] * 2.0 - 1.0)) + 1.0)
                     * a * p0m1[i];
          }

          if (kx > 0)
          {
            auto p0m1x = stdex::submdspan(P, idx(kx - 1, ky, kz),
                                          idx(0, 0, p - 1), stdex::full_extent);
            for (std::size_t i = 0; i < p00.size(); ++i)
              p00[i] += 2 * kx * a * p0m1x[i];
          }

          if (ky > 0)
          {
            auto p0m1y = stdex::submdspan(P, idx(kx, ky - 1, kz),
                                          idx(0, 0, p - 1), stdex::full_extent);
            for (std::size_t i = 0; i < p00.size(); ++i)
              p00[i] += ky * a * p0m1y[i];
          }

          if (kz > 0)
          {
            auto p0m1z = stdex::submdspan(P, idx(kx, ky, kz - 1),
                                          idx(0, 0, p - 1), stdex::full_extent);
            for (std::size_t i = 0; i < p00.size(); ++i)
              p00[i] += kz * a * p0m1z[i];
          }

          if (p > 1)
          {
            auto p0m2 = stdex::submdspan(P, idx(kx, ky, kz), idx(0, 0, p - 2),
                                         stdex::full_extent);
            for (std::size_t i = 0; i < p00.size(); ++i)
            {
              T f2 = x1[i] + x2[i] - 1.0;
              p00[i] -= f2 * f2 * p0m2[i] * (a - 1.0);
            }
            if (ky > 0)
            {
              auto p0m2y = stdex::submdspan(
                  P, idx(kx, ky - 1, kz), idx(0, 0, p - 2), stdex::full_extent);
              for (std::size_t i = 0; i < p00.size(); ++i)
              {
                p00[i] -= ky * ((x1[i] * 2.0 - 1.0) + (x2[i] * 2.0 - 1.0))
                          * p0m2y[i] * (a - 1.0);
              }
            }

            if (ky > 1)
            {
              auto p0m2y2 = stdex::submdspan(
                  P, idx(kx, ky - 2, kz), idx(0, 0, p - 2), stdex::full_extent);
              for (std::size_t i = 0; i < p00.size(); ++i)
                p00[i] -= ky * (ky - 1) * p0m2y2[i] * (a - 1.0);
            }

            if (kz > 0)
            {
              auto p0m2z = stdex::submdspan(
                  P, idx(kx, ky, kz - 1), idx(0, 0, p - 2), stdex::full_extent);
              for (std::size_t i = 0; i < p00.size(); ++i)
                p00[i] -= kz * ((x1[i] * 2.0 - 1.0) + (x2[i] * 2.0 - 1.0))
                          * p0m2z[i] * (a - 1.0);
            }

            if (kz > 1)
            {
              auto p0m2z2 = stdex::submdspan(
                  P, idx(kx, ky, kz - 2), idx(0, 0, p - 2), stdex::full_extent);
              for (std::size_t i = 0; i < p00.size(); ++i)
                p00[i] -= kz * (kz - 1) * p0m2z2[i] * (a - 1.0);
            }

            if (ky > 0 and kz > 0)
            {
              auto p0m2yz
                  = stdex::submdspan(P, idx(kx, ky - 1, kz - 1),
                                     idx(0, 0, p - 2), stdex::full_extent);
              for (std::size_t i = 0; i < p00.size(); ++i)
                p00[i] -= 2.0 * ky * kz * p0m2yz[i] * (a - 1.0);
            }
          }
        }

        for (std::size_t p = 0; p < n; ++p)
        {
          auto p10 = stdex::submdspan(P, idx(kx, ky, kz), idx(0, 1, p),
                                      stdex::full_extent);
          auto p00 = stdex::submdspan(P, idx(kx, ky, kz), idx(0, 0, p),
                                      stdex::full_extent);
          for (std::size_t i = 0; i < p10.size(); ++i)
            p10[i]
                = p00[i]
                  * ((1.0 + (x1[i] * 2.0 - 1.0)) * p
                     + (2.0 + (x1[i] * 2.0 - 1.0) * 3.0 + (x2[i] * 2.0 - 1.0))
                           * 0.5);
          if (ky > 0)
          {
            auto p0y = stdex::submdspan(P, idx(kx, ky - 1, kz), idx(0, 0, p),
                                        stdex::full_extent);
            for (std::size_t i = 0; i < p10.size(); ++i)
              p10[i] += 2 * ky * p0y[i] * (1.5 + p);
          }

          if (kz > 0)
          {
            auto p0z = stdex::submdspan(P, idx(kx, ky, kz - 1), idx(0, 0, p),
                                        stdex::full_extent);
            for (std::size_t i = 0; i < p10.size(); ++i)
              p10[i] += kz * p0z[i];
          }

          for (std::size_t q = 1; q < n - p; ++q)
          {
            auto [aq, bq, cq] = jrc<T>(2 * p + 1, q);
            auto pq1 = stdex::submdspan(P, idx(kx, ky, kz), idx(0, q + 1, p),
                                        stdex::full_extent);
            auto pq = stdex::submdspan(P, idx(kx, ky, kz), idx(0, q, p),
                                       stdex::full_extent);
            auto pqm1 = stdex::submdspan(P, idx(kx, ky, kz), idx(0, q - 1, p),
                                         stdex::full_extent);
            for (std::size_t i = 0; i < pq1.size(); ++i)
            {
              T f4 = 1.0 - x2[i];
              T f3 = (x1[i] * 2.0 - 1.0 + x2[i]);
              pq1[i] = pq[i] * (f3 * aq + f4 * bq) - pqm1[i] * f4 * f4 * cq;
            }
            if (ky > 0)
            {
              auto pqy = stdex::submdspan(P, idx(kx, ky - 1, kz), idx(0, q, p),
                                          stdex::full_extent);
              for (std::size_t i = 0; i < pq1.size(); ++i)
                pq1[i] += 2 * ky * pqy[i] * aq;
            }

            if (kz > 0)
            {
              auto pqz = stdex::submdspan(P, idx(kx, ky, kz - 1), idx(0, q, p),
                                          stdex::full_extent);
              auto pq1z = stdex::submdspan(
                  P, idx(kx, ky, kz - 1), idx(0, q - 1, p), stdex::full_extent);
              for (std::size_t i = 0; i < pq1.size(); ++i)
              {
                pq1[i] += kz * pqz[i] * (aq - bq)
                          + kz * (1.0 - (x2[i] * 2.0 - 1.0)) * pq1z[i] * cq;
              }
            }

            if (kz > 1)
            {
              auto pq1z2 = stdex::submdspan(
                  P, idx(kx, ky, kz - 2), idx(0, q - 1, p), stdex::full_extent);
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
            auto pq = stdex::submdspan(P, idx(kx, ky, kz), idx(1, q, p),
                                       stdex::full_extent);
            auto pq0 = stdex::submdspan(P, idx(kx, ky, kz), idx(0, q, p),
                                        stdex::full_extent);
            for (std::size_t i = 0; i < pq.size(); ++i)
            {
              pq[i] = pq0[i]
                      * ((1.0 + p + q) + (x2[i] * 2.0 - 1.0) * (2.0 + p + q));
            }

            if (kz > 0)
            {
              auto pqz = stdex::submdspan(P, idx(kx, ky, kz - 1), idx(0, q, p),
                                          stdex::full_extent);
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
              auto pqr1 = stdex::submdspan(P, idx(kx, ky, kz), idx(r + 1, q, p),
                                           stdex::full_extent);
              auto pqr = stdex::submdspan(P, idx(kx, ky, kz), idx(r, q, p),
                                          stdex::full_extent);
              auto pqrm1 = stdex::submdspan(
                  P, idx(kx, ky, kz), idx(r - 1, q, p), stdex::full_extent);

              for (std::size_t i = 0; i < pqr1.size(); ++i)
              {
                pqr1[i]
                    = pqr[i] * ((x2[i] * 2.0 - 1.0) * ar + br) - pqrm1[i] * cr;
              }

              if (kz > 0)
              {
                auto pqrz = stdex::submdspan(P, idx(kx, ky, kz - 1),
                                             idx(r, q, p), stdex::full_extent);
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
        auto pqr = stdex::submdspan(P, stdex::full_extent, idx(r, q, p),
                                    stdex::full_extent);
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
    stdex::mdspan<T, stdex::dextents<std::size_t, 3>> P, std::size_t n,
    std::size_t nderiv,
    stdex::mdspan<const T, stdex::dextents<std::size_t, 2>> x)
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

  const auto x0 = stdex::submdspan(x, stdex::full_extent, 0);
  const auto x1 = stdex::submdspan(x, stdex::full_extent, 1);
  const auto x2 = stdex::submdspan(x, stdex::full_extent, 2);

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
            auto p00 = stdex::submdspan(P, idx(kx, ky, kz), pyr_idx(p, 0, 0),
                                        stdex::full_extent);
            auto p1 = stdex::submdspan(P, idx(kx, ky, kz), pyr_idx(p - 1, 0, 0),
                                       stdex::full_extent);
            for (std::size_t i = 0; i < p00.size(); ++i)
              p00[i] = (0.5 + (x0[i] * 2.0 - 1.0) + (x2[i] * 2.0 - 1.0) * 0.5)
                       * p1[i] * (a + 1.0);

            if (kx > 0)
            {
              auto p11
                  = stdex::submdspan(P, idx(kx - 1, ky, kz),
                                     pyr_idx(p - 1, 0, 0), stdex::full_extent);

              for (std::size_t i = 0; i < p00.size(); ++i)
                p00[i] += 2.0 * kx * p11[i] * (a + 1.0);
            }

            if (kz > 0)
            {
              auto pz
                  = stdex::submdspan(P, idx(kx, ky, kz - 1),
                                     pyr_idx(p - 1, 0, 0), stdex::full_extent);

              for (std::size_t i = 0; i < p00.size(); ++i)
                p00[i] += kz * pz[i] * (a + 1.0);
            }

            if (p > 1)
            {
              auto p2 = stdex::submdspan(
                  P, idx(kx, ky, kz), pyr_idx(p - 2, 0, 0), stdex::full_extent);
              for (std::size_t i = 0; i < p00.size(); ++i)
              {
                T f2 = 1.0 - x2[i];
                p00[i] -= f2 * f2 * p2[i] * a;
              }

              if (kz > 0)
              {
                auto p2z = stdex::submdspan(P, idx(kx, ky, kz - 1),
                                            pyr_idx(p - 2, 0, 0),
                                            stdex::full_extent);
                for (std::size_t i = 0; i < p00.size(); ++i)
                  p00[i] += kz * (1.0 - (x2[i] * 2.0 - 1.0)) * p2z[i] * a;
              }

              if (kz > 1)
              {
                // quadratic term in z
                auto pz = stdex::submdspan(P, idx(kx, ky, kz - 2),
                                           pyr_idx(p - 2, 0, 0),
                                           stdex::full_extent);
                for (std::size_t i = 0; i < p00.size(); ++i)
                  p00[i] -= kz * (kz - 1) * pz[i] * a;
              }
            }
          }

          for (std::size_t q = 1; q < n + 1; ++q)
          {
            const T a = static_cast<T>(q - 1) / static_cast<T>(q);
            auto r_pq = stdex::submdspan(P, idx(kx, ky, kz), pyr_idx(p, q, 0),
                                         stdex::full_extent);

            auto _p = stdex::submdspan(P, idx(kx, ky, kz), pyr_idx(p, q - 1, 0),
                                       stdex::full_extent);
            for (std::size_t i = 0; i < r_pq.size(); ++i)
            {
              r_pq[i] = (0.5 + (x1[i] * 2.0 - 1.0) + (x2[i] * 2.0 - 1.0) * 0.5)
                        * _p[i] * (a + 1.0);
            }

            if (ky > 0)
            {
              auto _p
                  = stdex::submdspan(P, idx(kx, ky - 1, kz),
                                     pyr_idx(p, q - 1, 0), stdex::full_extent);
              for (std::size_t i = 0; i < r_pq.size(); ++i)
                r_pq[i] += 2.0 * ky * _p[i] * (a + 1.0);
            }

            if (kz > 0)
            {
              auto _p
                  = stdex::submdspan(P, idx(kx, ky, kz - 1),
                                     pyr_idx(p, q - 1, 0), stdex::full_extent);
              for (std::size_t i = 0; i < r_pq.size(); ++i)
                r_pq[i] += kz * _p[i] * (a + 1.0);
            }

            if (q > 1)
            {
              auto _p = stdex::submdspan(
                  P, idx(kx, ky, kz), pyr_idx(p, q - 2, 0), stdex::full_extent);
              for (std::size_t i = 0; i < r_pq.size(); ++i)
              {
                const T f2 = 1.0 - x2[i];
                r_pq[i] -= f2 * f2 * _p[i] * a;
              }

              if (kz > 0)
              {
                auto _p = stdex::submdspan(P, idx(kx, ky, kz - 1),
                                           pyr_idx(p, q - 2, 0),
                                           stdex::full_extent);
                for (std::size_t i = 0; i < r_pq.size(); ++i)
                  r_pq[i] += kz * (1.0 - (x2[i] * 2.0 - 1.0)) * _p[i] * a;
              }

              if (kz > 1)
              {
                auto _p = stdex::submdspan(P, idx(kx, ky, kz - 2),
                                           pyr_idx(p, q - 2, 0),
                                           stdex::full_extent);
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
            auto r_pq1 = stdex::submdspan(P, idx(kx, ky, kz), pyr_idx(p, q, 1),
                                          stdex::full_extent);

            auto r_pq0 = stdex::submdspan(P, idx(kx, ky, kz), pyr_idx(p, q, 0),
                                          stdex::full_extent);
            for (std::size_t i = 0; i < r_pq1.size(); ++i)
            {
              r_pq1[i]
                  = r_pq0[i]
                    * ((1.0 + p + q) + (x2[i] * 2.0 - 1.0) * (2.0 + p + q));
            }

            if (kz > 0)
            {
              auto r_pq = stdex::submdspan(
                  P, idx(kx, ky, kz - 1), pyr_idx(p, q, 0), stdex::full_extent);
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
              auto r_pqr = stdex::submdspan(
                  P, idx(kx, ky, kz), pyr_idx(p, q, r + 1), stdex::full_extent);
              auto _r0 = stdex::submdspan(P, idx(kx, ky, kz), pyr_idx(p, q, r),
                                          stdex::full_extent);
              auto _r1 = stdex::submdspan(
                  P, idx(kx, ky, kz), pyr_idx(p, q, r - 1), stdex::full_extent);
              for (std::size_t i = 0; i < r_pqr.size(); ++i)
              {
                r_pqr[i]
                    = _r0[i] * ((x2[i] * 2.0 - 1.0) * ar + br) - _r1[i] * cr;
              }

              if (kz > 0)
              {
                auto _r
                    = stdex::submdspan(P, idx(kx, ky, kz - 1), pyr_idx(p, q, r),
                                       stdex::full_extent);
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
        auto pqr = stdex::submdspan(P, stdex::full_extent, pyr_idx(p, q, r),
                                    stdex::full_extent);
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
    stdex::mdspan<T, stdex::dextents<std::size_t, 3>> P, std::size_t n,
    std::size_t nderiv,
    stdex::mdspan<const T, stdex::dextents<std::size_t, 2>> x)
{
  assert(x.extent(1) == 2);
  assert(P.extent(0) == (nderiv + 1) * (nderiv + 2) / 2);
  assert(P.extent(1) == (n + 1) * (n + 1));
  assert(P.extent(2) == x.extent(0));

  // Indexing for quadrilateral basis functions
  auto quad_idx = [n](std::size_t px, std::size_t py) -> std::size_t
  { return (n + 1) * px + py; };

  // Compute 1D basis
  const auto x0 = stdex::submdspan(x, stdex::full_extent, 0);
  const auto x1 = stdex::submdspan(x, stdex::full_extent, 1);

  assert(x0.extent(0) > 0);
  assert(x1.extent(0) > 0);

  // Compute tabulation of interval for px = 0
  std::fill(P.data_handle(), P.data_handle() + P.size(), 0.0);
  for (std::size_t j = 0; j < P.extent(2); ++j)
    P(idx(0, 0), quad_idx(0, 0), j) = 1.0;

  if (n == 0)
    return;

  { // scope
    auto result = stdex::submdspan(P, idx(0, 0), stdex::full_extent,
                                   stdex::full_extent);
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
    auto result = stdex::submdspan(P, idx(0, ky), stdex::full_extent,
                                   stdex::full_extent);
    auto result0 = stdex::submdspan(P, idx(0, ky - 1), stdex::full_extent,
                                    stdex::full_extent);
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
    auto result = stdex::submdspan(P, idx(0, ky), stdex::full_extent,
                                   stdex::full_extent);
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
      auto result = stdex::submdspan(P, idx(0, ky), stdex::full_extent,
                                     stdex::full_extent);
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
      auto result = stdex::submdspan(P, idx(kx, ky), stdex::full_extent,
                                     stdex::full_extent);
      auto result0 = stdex::submdspan(P, idx(kx - 1, ky), stdex::full_extent,
                                      stdex::full_extent);
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
        auto result = stdex::submdspan(P, idx(kx, ky), stdex::full_extent,
                                       stdex::full_extent);
        auto result0 = stdex::submdspan(P, idx(kx - 1, ky), stdex::full_extent,
                                        stdex::full_extent);
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
      auto pxy = stdex::submdspan(P, stdex::full_extent, quad_idx(px, py),
                                  stdex::full_extent);
      for (std::size_t i = 0; i < pxy.extent(0); ++i)
        for (std::size_t j = 0; j < pxy.extent(1); ++j)
          pxy(i, j) *= std::sqrt((2 * px + 1) * (2 * py + 1));
    }
  }
}
//-----------------------------------------------------------------------------
template <typename T>
void tabulate_polyset_hex_derivs(
    stdex::mdspan<T, stdex::dextents<std::size_t, 3>> P, std::size_t n,
    std::size_t nderiv,
    stdex::mdspan<const T, stdex::dextents<std::size_t, 2>> x)
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
  const auto x0 = stdex::submdspan(x, stdex::full_extent, 0);
  const auto x1 = stdex::submdspan(x, stdex::full_extent, 1);
  const auto x2 = stdex::submdspan(x, stdex::full_extent, 2);
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
    auto result = stdex::submdspan(P, idx(0, 0, 0), stdex::full_extent,
                                   stdex::full_extent);
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
    auto result = stdex::submdspan(P, idx(0, 0, kz), stdex::full_extent,
                                   stdex::full_extent);
    auto result0 = stdex::submdspan(P, idx(0, 0, kz - 1), stdex::full_extent,
                                    stdex::full_extent);
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
    auto result = stdex::submdspan(P, idx(0, 0, kz), stdex::full_extent,
                                   stdex::full_extent);
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
      auto result = stdex::submdspan(P, idx(0, 0, kz), stdex::full_extent,
                                     stdex::full_extent);
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
      auto result = stdex::submdspan(P, idx(0, ky, kz), stdex::full_extent,
                                     stdex::full_extent);
      auto result0 = stdex::submdspan(P, idx(0, ky - 1, kz), stdex::full_extent,
                                      stdex::full_extent);
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
        auto result = stdex::submdspan(P, idx(0, ky, kz), stdex::full_extent,
                                       stdex::full_extent);
        auto result0 = stdex::submdspan(P, idx(0, ky - 1, kz),
                                        stdex::full_extent, stdex::full_extent);
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
      auto result = stdex::submdspan(P, idx(0, ky, kz), stdex::full_extent,
                                     stdex::full_extent);
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
        auto result = stdex::submdspan(P, idx(0, ky, kz), stdex::full_extent,
                                       stdex::full_extent);
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
          auto result = stdex::submdspan(P, idx(kx, ky, kz), stdex::full_extent,
                                         stdex::full_extent);
          auto result0 = stdex::submdspan(
              P, idx(kx - 1, ky, kz), stdex::full_extent, stdex::full_extent);
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
          auto result = stdex::submdspan(P, idx(kx, ky, kz), stdex::full_extent,
                                         stdex::full_extent);
          auto result0 = stdex::submdspan(
              P, idx(kx - 1, ky, kz), stdex::full_extent, stdex::full_extent);
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
        auto pxyz = stdex::submdspan(P, stdex::full_extent, hex_idx(px, py, pz),
                                     stdex::full_extent);
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
    stdex::mdspan<T, stdex::dextents<std::size_t, 3>> P, std::size_t n,
    std::size_t nderiv,
    stdex::mdspan<const T, stdex::dextents<std::size_t, 2>> x)
{
  assert(x.extent(1) == 3);
  assert(P.extent(0) == (nderiv + 1) * (nderiv + 2) * (nderiv + 3) / 6);
  assert(P.extent(1) == (n + 1) * (n + 1) * (n + 2) / 2);
  assert(P.extent(2) == x.extent(0));

  const auto x0 = stdex::submdspan(x, stdex::full_extent, 0);
  const auto x1 = stdex::submdspan(x, stdex::full_extent, 1);
  const auto x2 = stdex::submdspan(x, stdex::full_extent, 2);

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
        auto p0 = stdex::submdspan(P, idx(kx, ky, 0), prism_idx(p, 0, 0),
                                   stdex::full_extent);
        auto p1 = stdex::submdspan(P, idx(kx, ky, 0), prism_idx(p - 1, 0, 0),
                                   stdex::full_extent);
        const T a = static_cast<T>(2 * p - 1) / static_cast<T>(p);
        for (std::size_t i = 0; i < p0.size(); ++i)
        {
          p0[i] = ((x0[i] * 2.0 - 1.0) + 0.5 * (x1[i] * 2.0 - 1.0) + 0.5)
                  * p1[i] * a;
        }

        if (kx > 0)
        {
          auto result0
              = stdex::submdspan(P, idx(kx - 1, ky, 0), prism_idx(p - 1, 0, 0),
                                 stdex::full_extent);
          for (std::size_t i = 0; i < p0.size(); ++i)
            p0[i] += 2 * kx * a * result0[i];
        }

        if (ky > 0)
        {
          auto result0
              = stdex::submdspan(P, idx(kx, ky - 1, 0), prism_idx(p - 1, 0, 0),
                                 stdex::full_extent);
          for (std::size_t i = 0; i < p0.size(); ++i)
            p0[i] += ky * a * result0[i];
        }

        if (p > 1)
        {
          // y^2 terms
          auto p2 = stdex::submdspan(P, idx(kx, ky, 0), prism_idx(p - 2, 0, 0),
                                     stdex::full_extent);
          for (std::size_t i = 0; i < p0.size(); ++i)
          {
            T f2 = (1.0 - x1[i]);
            p0[i] -= f2 * f2 * p2[i] * (a - 1.0);
          }

          if (ky > 0)
          {
            auto result0
                = stdex::submdspan(P, idx(kx, ky - 1, 0),
                                   prism_idx(p - 2, 0, 0), stdex::full_extent);
            for (std::size_t i = 0; i < p0.size(); ++i)
            {
              p0[i]
                  -= ky * ((x1[i] * 2.0 - 1.0) - 1.0) * result0[i] * (a - 1.0);
            }
          }

          if (ky > 1)
          {
            auto result0
                = stdex::submdspan(P, idx(kx, ky - 2, 0),
                                   prism_idx(p - 2, 0, 0), stdex::full_extent);
            for (std::size_t i = 0; i < p0.size(); ++i)
              p0[i] -= ky * (ky - 1) * result0[i] * (a - 1.0);
          }
        }
      }

      for (std::size_t p = 0; p < n; ++p)
      {
        auto p0 = stdex::submdspan(P, idx(kx, ky, 0), prism_idx(p, 0, 0),
                                   stdex::full_extent);
        auto p1 = stdex::submdspan(P, idx(kx, ky, 0), prism_idx(p, 1, 0),
                                   stdex::full_extent);
        for (std::size_t i = 0; i < p1.size(); ++i)
          p1[i] = p0[i] * ((x1[i] * 2.0 - 1.0) * (1.5 + p) + 0.5 + p);

        if (ky > 0)
        {
          auto result0 = stdex::submdspan(
              P, idx(kx, ky - 1, 0), prism_idx(p, 0, 0), stdex::full_extent);
          for (std::size_t i = 0; i < p1.size(); ++i)
            p1[i] += 2 * ky * (1.5 + p) * result0[i];
        }

        for (std::size_t q = 1; q < n - p; ++q)
        {
          auto pqp1 = stdex::submdspan(
              P, idx(kx, ky, 0), prism_idx(p, q + 1, 0), stdex::full_extent);
          auto pq = stdex::submdspan(P, idx(kx, ky, 0), prism_idx(p, q, 0),
                                     stdex::full_extent);
          auto pqm1 = stdex::submdspan(
              P, idx(kx, ky, 0), prism_idx(p, q - 1, 0), stdex::full_extent);
          const auto [a1, a2, a3] = jrc<T>(2 * p + 1, q);
          for (std::size_t i = 0; i < p0.size(); ++i)
            pqp1[i] = pq[i] * ((x1[i] * 2.0 - 1.0) * a1 + a2) - pqm1[i] * a3;

          if (ky > 0)
          {
            auto result0 = stdex::submdspan(
                P, idx(kx, ky - 1, 0), prism_idx(p, q, 0), stdex::full_extent);
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
          auto result = stdex::submdspan(P, idx(kx, ky, kz), stdex::full_extent,
                                         stdex::full_extent);
          auto result0 = stdex::submdspan(
              P, idx(kx, ky, kz - 1), stdex::full_extent, stdex::full_extent);
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
        auto pqr = stdex::submdspan(P, stdex::full_extent, prism_idx(p, q, r),
                                    stdex::full_extent);
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
    std::experimental::mdspan<T, std::experimental::dextents<std::size_t, 3>> P,
    cell::type celltype, int d, int n,
    std::experimental::mdspan<const T,
                              std::experimental::dextents<std::size_t, 2>>
        x)
{
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
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 3>> polyset::tabulate(
    cell::type celltype, int d, int n,
    std::experimental::mdspan<const T,
                              std::experimental::dextents<std::size_t, 2>>
        x)
{
  std::array<std::size_t, 3> shape
      = {(std::size_t)polyset::nderivs(celltype, n),
         (std::size_t)polyset::dim(celltype, d), x.extent(0)};
  std::vector<T> P(shape[0] * shape[1] * shape[2]);
  stdex::mdspan<T, stdex::dextents<std::size_t, 3>> _P(P.data(), shape);
  polyset::tabulate(_P, celltype, d, n, x);
  return {std::move(P), std::move(shape)};
}
//-----------------------------------------------------------------------------
/// @cond
template std::pair<std::vector<float>, std::array<std::size_t, 3>>
polyset::tabulate(
    cell::type, int, int,
    std::experimental::mdspan<const float,
                              std::experimental::dextents<std::size_t, 2>>);
template std::pair<std::vector<double>, std::array<std::size_t, 3>>
polyset::tabulate(
    cell::type, int, int,
    std::experimental::mdspan<const double,
                              std::experimental::dextents<std::size_t, 2>>);
/// @endcond
//-----------------------------------------------------------------------------
int polyset::dim(cell::type celltype, int d)
{
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
