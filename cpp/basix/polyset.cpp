// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "polyset.h"
#include "cell.h"
#include "indexing.h"
#include <array>
#include <cmath>
#include <stdexcept>
#include <xtensor/xview.hpp>

using namespace basix;
using namespace basix::indexing;

namespace
{
// Compute coefficients in the Jacobi Polynomial recurrence relation
constexpr std::array<double, 3> jrc(int a, int n)
{
  double an = (a + 2 * n + 1) * (a + 2 * n + 2)
              / static_cast<double>(2 * (n + 1) * (a + n + 1));
  double bn = a * a * (a + 2 * n + 1)
              / static_cast<double>(2 * (n + 1) * (a + n + 1) * (a + 2 * n));
  double cn = n * (a + n) * (a + 2 * n + 2)
              / static_cast<double>((n + 1) * (a + n + 1) * (a + 2 * n));
  return {an, bn, cn};
}
//-----------------------------------------------------------------------------
// At a point, only the constant polynomial can be used. This has value 1 and
// derivative 0.
void tabulate_polyset_point_derivs(xt::xtensor<double, 3>& P, std::size_t,
                                   std::size_t nderiv,
                                   const xt::xtensor<double, 2>& x)
{
  assert(x.shape(0) > 0);
  assert(P.shape(0) == nderiv + 1);
  assert(P.shape(1) == x.shape(0));
  assert(P.shape(2) == 1);

  xt::view(P, xt::all(), xt::all(), 0) = 0.0;
  xt::view(P, 0, xt::all(), 0) = 1.0;
}
//-----------------------------------------------------------------------------
// Compute the complete set of derivatives from 0 to nderiv, for all the
// polynomials up to order n on a line segment. The polynomials used are
// Legendre Polynomials, with the recurrence relation given by
// n P(n) = (2n - 1) x P_{n-1} - (n - 1) P_{n-2} in the interval [-1, 1]. The
// range is rescaled here to [0, 1].
void tabulate_polyset_line_derivs(xt::xtensor<double, 3>& P, std::size_t degree,
                                  std::size_t nderiv,
                                  const xt::xtensor<double, 1>& x)
{
  assert(x.shape(0) > 0);
  const auto X = x * 2.0 - 1.0;
  const std::size_t m = (degree + 1);
  assert(P.shape(0) == nderiv + 1);
  assert(P.shape(1) == x.shape(0));
  assert(P.shape(2) == m);

  xt::view(P, xt::all(), xt::all(), 0) = 0.0;
  xt::view(P, 0, xt::all(), 0) = 1.0;

  for (std::size_t k = 0; k <= nderiv; ++k)
  {
    // Get reference to this derivative
    auto result = xt::view(P, k, xt::all(), xt::all());
    auto result0 = xt::view(P, k - 1, xt::all(), xt::all());
    for (std::size_t p = 1; p <= degree; ++p)
    {
      const double a = 1.0 - 1.0 / static_cast<double>(p);
      xt::col(result, p) = X * xt::col(result, p - 1) * (a + 1.0);
      if (k > 0)
        xt::col(result, p) += 2 * k * xt::col(result0, p - 1) * (a + 1.0);
      if (p > 1)
        xt::col(result, p) -= xt::col(result, p - 2) * a;
    }
  }

  // Normalise
  for (std::size_t p = 0; p <= degree; ++p)
    xt::view(P, xt::all(), xt::all(), p) *= std::sqrt(2 * p + 1);
}
//-----------------------------------------------------------------------------
// Compute the complete set of derivatives from 0 to nderiv, for all the
// polynomials up to order n on a triangle in [0, 1][0, 1]. The
// polynomials P_{pq} are built up in sequence, firstly along q = 0,
// which is a line segment, as in tabulate_polyset_interval_derivs
// above, but with a change of variables. The polynomials are then
// extended in the q direction, using the relation given in Sherwin and
// Karniadakis 1995 (https://doi.org/10.1016/0045-7825(94)00745-9).
void tabulate_polyset_triangle_derivs(xt::xtensor<double, 3>& P, std::size_t n,
                                      std::size_t nderiv,
                                      const xt::xtensor<double, 2>& pts)
{
  assert(pts.shape(1) == 2);

  const auto x = pts * 2.0 - 1.0;
  auto x0 = xt::col(x, 0);
  auto x1 = xt::col(x, 1);

  const std::size_t m = (n + 1) * (n + 2) / 2;
  const std::size_t md = (nderiv + 1) * (nderiv + 2) / 2;
  assert(P.shape(0) == md);
  assert(P.shape(1) == x.shape(0));
  assert(P.shape(2) == m);

  // f3 = ((1 - y) / 2)^2
  const auto f3 = xt::square(1.0 - x1) * 0.25;

  xt::view(P, xt::all(), xt::all(), 0) = 0.0;
  xt::view(P, idx(0, 0), xt::all(), 0) = 1.0;

  // Iterate over derivatives in increasing order, since higher derivatives
  for (std::size_t kx = 0; kx <= nderiv; ++kx)
  {
    for (std::size_t ky = 0; ky <= nderiv - kx; ++ky)
    {
      for (std::size_t p = 1; p <= n; ++p)
      {
        auto p0 = xt::view(P, idx(kx, ky), xt::all(), idx(p, 0));
        const double a
            = static_cast<double>(2 * p - 1) / static_cast<double>(p);
        p0 = (x0 + 0.5 * x1 + 0.5)
             * xt::view(P, idx(kx, ky), xt::all(), idx(p - 1, 0)) * a;
        if (kx > 0)
        {
          auto result0 = xt::view(P, idx(kx - 1, ky), xt::all(), idx(p - 1, 0));
          p0 += 2 * kx * a * result0;
        }

        if (ky > 0)
        {
          auto result0 = xt::view(P, idx(kx, ky - 1), xt::all(), idx(p - 1, 0));
          p0 += ky * a * result0;
        }

        if (p > 1)
        {
          // y^2 terms
          p0 -= f3 * xt::view(P, idx(kx, ky), xt::all(), idx(p - 2, 0))
                * (a - 1.0);
          if (ky > 0)
          {
            auto result0
                = xt::view(P, idx(kx, ky - 1), xt::all(), idx(p - 2, 0));
            p0 -= ky * (x1 - 1.0) * result0 * (a - 1.0);
          }

          if (ky > 1)
          {
            auto result0
                = xt::view(P, idx(kx, ky - 2), xt::all(), idx(p - 2, 0));
            p0 -= ky * (ky - 1) * result0 * (a - 1.0);
          }
        }
      }

      for (std::size_t p = 0; p < n; ++p)
      {
        auto p0 = xt::view(P, idx(kx, ky), xt::all(), idx(p, 0));
        auto p1 = xt::view(P, idx(kx, ky), xt::all(), idx(p, 1));
        p1 = p0 * (x1 * (1.5 + p) + 0.5 + p);
        if (ky > 0)
        {
          auto result0 = xt::view(P, idx(kx, ky - 1), xt::all(), idx(p, 0));
          p1 += 2 * ky * (1.5 + p) * result0;
        }

        for (std::size_t q = 1; q < n - p; ++q)
        {
          const auto [a1, a2, a3] = jrc(2 * p + 1, q);
          xt::view(P, idx(kx, ky), xt::all(), idx(p, q + 1))
              = xt::view(P, idx(kx, ky), xt::all(), idx(p, q)) * (x1 * a1 + a2)
                - xt::view(P, idx(kx, ky), xt::all(), idx(p, q - 1)) * a3;
          if (ky > 0)
          {
            auto result0 = xt::view(P, idx(kx, ky - 1), xt::all(), idx(p, q));
            xt::view(P, idx(kx, ky), xt::all(), idx(p, q + 1))
                += 2 * ky * a1 * result0;
          }
        }
      }
    }
  }

  // Normalisation
  for (std::size_t p = 0; p <= n; ++p)
    for (std::size_t q = 0; q <= n - p; ++q)
      xt::view(P, xt::all(), xt::all(), idx(p, q))
          *= std::sqrt((p + 0.5) * (p + q + 1)) * 2;
}
//-----------------------------------------------------------------------------
void tabulate_polyset_tetrahedron_derivs(xt::xtensor<double, 3>& P,
                                         std::size_t n, std::size_t nderiv,
                                         const xt::xtensor<double, 2>& pts)
{
  assert(pts.shape(1) == 3);
  const std::size_t m = (n + 1) * (n + 2) * (n + 3) / 6;
  const std::size_t md = (nderiv + 1) * (nderiv + 2) * (nderiv + 3) / 6;

  auto x = pts * 2.0 - 1.0;
  const auto x0 = xt::col(x, 0);
  const auto x1 = xt::col(x, 1);
  const auto x2 = xt::col(x, 2);

  auto f2 = 0.25 * xt::square(x1 + x2);
  auto f3 = 0.5 * (1.0 + x1 * 2.0 + x2);
  auto f4 = 0.5 * (1.0 - x2);
  auto f5 = f4 * f4;

  // Traverse derivatives in increasing order
  assert(P.shape(0) == md);
  assert(P.shape(1) == pts.shape(0));
  assert(P.shape(2) == m);

  xt::view(P, xt::all(), xt::all(), 0) = 0.0;
  xt::view(P, idx(0, 0, 0), xt::all(), 0) = 1.0;

  for (std::size_t kx = 0; kx <= nderiv; ++kx)
  {
    for (std::size_t ky = 0; ky <= nderiv - kx; ++ky)
    {
      for (std::size_t kz = 0; kz <= nderiv - kx - ky; ++kz)
      {
        for (std::size_t p = 1; p <= n; ++p)
        {
          auto p00 = xt::view(P, idx(kx, ky, kz), xt::all(), idx(p, 0, 0));
          double a = static_cast<double>(2 * p - 1) / static_cast<double>(p);
          p00 = (x0 + 0.5 * (x1 + x2) + 1.0)
                * xt::view(P, idx(kx, ky, kz), xt::all(), idx(p - 1, 0, 0)) * a;
          if (kx > 0)
          {
            p00 += 2 * kx * a
                   * xt::view(P, idx(kx - 1, ky, kz), xt::all(),
                              idx(p - 1, 0, 0));
          }

          if (ky > 0)
          {
            p00 += ky * a
                   * xt::view(P, idx(kx, ky - 1, kz), xt::all(),
                              idx(p - 1, 0, 0));
          }

          if (kz > 0)
          {
            p00 += kz * a
                   * xt::view(P, idx(kx, ky, kz - 1), xt::all(),
                              idx(p - 1, 0, 0));
          }

          if (p > 1)
          {
            p00 -= f2
                   * xt::view(P, idx(kx, ky, kz), xt::all(), idx(p - 2, 0, 0))
                   * (a - 1.0);
            if (ky > 0)
            {
              p00 -= ky * (x1 + x2)
                     * xt::view(P, idx(kx, ky - 1, kz), xt::all(),
                                idx(p - 2, 0, 0))
                     * (a - 1.0);
            }

            if (ky > 1)
            {
              p00 -= ky * (ky - 1)
                     * xt::view(P, idx(kx, ky - 2, kz), xt::all(),
                                idx(p - 2, 0, 0))
                     * (a - 1.0);
            }

            if (kz > 0)
            {
              p00 -= kz * (x1 + x2)
                     * xt::view(P, idx(kx, ky, kz - 1), xt::all(),
                                idx(p - 2, 0, 0))
                     * (a - 1.0);
            }

            if (kz > 1)
            {
              p00 -= kz * (kz - 1)
                     * xt::view(P, idx(kx, ky, kz - 2), xt::all(),
                                idx(p - 2, 0, 0))
                     * (a - 1.0);
            }

            if (ky > 0 and kz > 0)
            {
              p00 -= 2.0 * ky * kz
                     * xt::view(P, idx(kx, ky - 1, kz - 1), xt::all(),
                                idx(p - 2, 0, 0))
                     * (a - 1.0);
            }
          }
        }

        for (std::size_t p = 0; p < n; ++p)
        {
          auto p10 = xt::view(P, idx(kx, ky, kz), xt::all(), idx(p, 1, 0));
          p10 = xt::view(P, idx(kx, ky, kz), xt::all(), idx(p, 0, 0))
                * ((1.0 + x1) * p + (2.0 + x1 * 3.0 + x2) * 0.5);
          if (ky > 0)
          {
            p10 += 2 * ky
                   * xt::view(P, idx(kx, ky - 1, kz), xt::all(), idx(p, 0, 0))
                   * (1.5 + p);
          }

          if (kz > 0)
          {
            p10 += kz
                   * xt::view(P, idx(kx, ky, kz - 1), xt::all(), idx(p, 0, 0));
          }

          for (std::size_t q = 1; q < n - p; ++q)
          {
            auto [aq, bq, cq] = jrc(2 * p + 1, q);
            auto pq1
                = xt::view(P, idx(kx, ky, kz), xt::all(), idx(p, q + 1, 0));
            pq1 = xt::view(P, idx(kx, ky, kz), xt::all(), idx(p, q, 0))
                      * (f3 * aq + f4 * bq)
                  - xt::view(P, idx(kx, ky, kz), xt::all(), idx(p, q - 1, 0))
                        * f5 * cq;

            if (ky > 0)
            {
              pq1 += 2 * ky
                     * xt::view(P, idx(kx, ky - 1, kz), xt::all(), idx(p, q, 0))
                     * aq;
            }

            if (kz > 0)
            {
              pq1 += kz
                         * xt::view(P, idx(kx, ky, kz - 1), xt::all(),
                                    idx(p, q, 0))
                         * (aq - bq)
                     + kz * (1.0 - x2)
                           * xt::view(P, idx(kx, ky, kz - 1), xt::all(),
                                      idx(p, q - 1, 0))
                           * cq;
            }

            if (kz > 1)
            {
              // Quadratic term in z
              pq1 -= kz * (kz - 1)
                     * xt::view(P, idx(kx, ky, kz - 2), xt::all(),
                                idx(p, q - 1, 0))
                     * cq;
            }
          }
        }

        for (std::size_t p = 0; p < n; ++p)
        {
          for (std::size_t q = 0; q < n - p; ++q)
          {
            auto pq = xt::view(P, idx(kx, ky, kz), xt::all(), idx(p, q, 1));
            pq = xt::view(P, idx(kx, ky, kz), xt::all(), idx(p, q, 0))
                 * ((1.0 + p + q) + x2 * (2.0 + p + q));
            if (kz > 0)
            {
              pq += 2 * kz * (2.0 + p + q)
                    * xt::view(P, idx(kx, ky, kz - 1), xt::all(), idx(p, q, 0));
            }
          }
        }

        for (std::size_t p = 0; p + 1 < n; ++p)
        {
          for (std::size_t q = 0; q + 1 < n - p; ++q)
          {
            for (std::size_t r = 1; r < n - p - q; ++r)
            {
              auto [ar, br, cr] = jrc(2 * p + 2 * q + 2, r);
              xt::view(P, idx(kx, ky, kz), xt::all(), idx(p, q, r + 1))
                  = xt::view(P, idx(kx, ky, kz), xt::all(), idx(p, q, r))
                        * (x2 * ar + br)
                    - xt::view(P, idx(kx, ky, kz), xt::all(), idx(p, q, r - 1))
                          * cr;
              if (kz > 0)
              {
                xt::view(P, idx(kx, ky, kz), xt::all(), idx(p, q, r + 1))
                    += 2 * kz * ar
                       * xt::view(P, idx(kx, ky, kz - 1), xt::all(),
                                  idx(p, q, r));
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
        xt::view(P, xt::all(), xt::all(), idx(p, q, r))
            *= std::sqrt(2 * (p + 0.5) * (p + q + 1.0) * (p + q + r + 1.5)) * 2;
      }
    }
  }
}
//-----------------------------------------------------------------------------
void tabulate_polyset_pyramid_derivs(xt::xtensor<double, 3>& P, std::size_t n,
                                     std::size_t nderiv,
                                     const xt::xtensor<double, 2>& pts)
{
  assert(pts.shape(1) == 3);
  const std::size_t m = (n + 1) * (n + 2) * (2 * n + 3) / 6;
  const std::size_t md = (nderiv + 1) * (nderiv + 2) * (nderiv + 3) / 6;

  // Indexing for pyramidal basis functions
  auto pyr_idx = [n](std::size_t p, std::size_t q, std::size_t r) -> std::size_t
  {
    const std::size_t rv = n - r + 1;
    const std::size_t r0
        = r * (n + 1) * (n - r + 2) + (2 * r - 1) * (r - 1) * r / 6;
    return r0 + p * rv + q;
  };

  auto x = pts * 2.0 - 1.0;
  const auto x0 = xt::col(x, 0);
  const auto x1 = xt::col(x, 1);
  const auto x2 = xt::col(x, 2);

  auto f2 = 0.25 * xt::square(1.0 - x2);

  // Traverse derivatives in increasing order
  assert(P.shape(0) == md);
  assert(P.shape(1) == pts.shape(0));
  assert(P.shape(2) == m);

  xt::view(P, xt::all(), xt::all(), pyr_idx(0, 0, 0)) = 0.0;
  xt::view(P, idx(0, 0, 0), xt::all(), pyr_idx(0, 0, 0)) = 1.0;

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
            const double a
                = static_cast<double>(p - 1) / static_cast<double>(p);
            auto p00
                = xt::view(P, idx(kx, ky, kz), xt::all(), pyr_idx(p, 0, 0));
            p00 = (0.5 + x0 + x2 * 0.5)
                  * xt::view(P, idx(kx, ky, kz), xt::all(),
                             pyr_idx(p - 1, 0, 0))
                  * (a + 1.0);

            if (kx > 0)
            {
              p00 += 2.0 * kx
                     * xt::view(P, idx(kx - 1, ky, kz), xt::all(),
                                pyr_idx(p - 1, 0, 0))
                     * (a + 1.0);
            }

            if (kz > 0)
            {
              p00 += kz
                     * xt::view(P, idx(kx, ky, kz - 1), xt::all(),
                                pyr_idx(p - 1, 0, 0))
                     * (a + 1.0);
            }

            if (p > 1)
            {
              p00 -= f2
                     * xt::view(P, idx(kx, ky, kz), xt::all(),
                                pyr_idx(p - 2, 0, 0))
                     * a;

              if (kz > 0)
              {
                p00 += kz * (1.0 - x2)
                       * xt::view(P, idx(kx, ky, kz - 1), xt::all(),
                                  pyr_idx(p - 2, 0, 0))
                       * a;
              }

              if (kz > 1)
              {
                // quadratic term in z
                p00 -= kz * (kz - 1)
                       * xt::view(P, idx(kx, ky, kz - 2), xt::all(),
                                  pyr_idx(p - 2, 0, 0))
                       * a;
              }
            }
          }

          for (std::size_t q = 1; q < n + 1; ++q)
          {
            const double a
                = static_cast<double>(q - 1) / static_cast<double>(q);
            auto r_pq
                = xt::view(P, idx(kx, ky, kz), xt::all(), pyr_idx(p, q, 0));
            r_pq = (0.5 + x1 + x2 * 0.5)
                   * xt::view(P, idx(kx, ky, kz), xt::all(),
                              pyr_idx(p, q - 1, 0))
                   * (a + 1.0);
            if (ky > 0)
            {
              r_pq += 2.0 * ky
                      * xt::view(P, idx(kx, ky - 1, kz), xt::all(),
                                 pyr_idx(p, q - 1, 0))
                      * (a + 1.0);
            }

            if (kz > 0)
            {
              r_pq += kz
                      * xt::view(P, idx(kx, ky, kz - 1), xt::all(),
                                 pyr_idx(p, q - 1, 0))
                      * (a + 1.0);
            }

            if (q > 1)
            {
              r_pq -= f2
                      * xt::view(P, idx(kx, ky, kz), xt::all(),
                                 pyr_idx(p, q - 2, 0))
                      * a;

              if (kz > 0)
              {
                r_pq += kz * (1.0 - x2)
                        * xt::view(P, idx(kx, ky, kz - 1), xt::all(),
                                   pyr_idx(p, q - 2, 0))
                        * a;
              }

              if (kz > 1)
              {
                r_pq -= kz * (kz - 1)
                        * xt::view(P, idx(kx, ky, kz - 2), xt::all(),
                                   pyr_idx(p, q - 2, 0))
                        * a;
              }
            }
          }
        }

        // Extend into r > 0
        for (std::size_t p = 0; p < n; ++p)
        {
          for (std::size_t q = 0; q < n; ++q)
          {
            auto r_pq1
                = xt::view(P, idx(kx, ky, kz), xt::all(), pyr_idx(p, q, 1));
            r_pq1 = xt::view(P, idx(kx, ky, kz), xt::all(), pyr_idx(p, q, 0))
                    * ((1.0 + p + q) + x2 * (2.0 + p + q));
            if (kz > 0)
            {
              r_pq1 += 2 * kz
                       * xt::view(P, idx(kx, ky, kz - 1), xt::all(),
                                  pyr_idx(p, q, 0))
                       * (2.0 + p + q);
            }
          }
        }

        for (std::size_t r = 1; r <= n; ++r)
        {
          for (std::size_t p = 0; p < n - r; ++p)
          {
            for (std::size_t q = 0; q < n - r; ++q)
            {
              auto [ar, br, cr] = jrc(2 * p + 2 * q + 2, r);
              auto r_pqr = xt::view(P, idx(kx, ky, kz), xt::all(),
                                    pyr_idx(p, q, r + 1));
              r_pqr = xt::view(P, idx(kx, ky, kz), xt::all(), pyr_idx(p, q, r))
                          * (x2 * ar + br)
                      - xt::view(P, idx(kx, ky, kz), xt::all(),
                                 pyr_idx(p, q, r - 1))
                            * cr;
              if (kz > 0)
              {
                r_pqr += ar * 2 * kz
                         * xt::view(P, idx(kx, ky, kz - 1), xt::all(),
                                    pyr_idx(p, q, r));
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
        xt::view(P, xt::all(), xt::all(), pyr_idx(p, q, r))
            *= std::sqrt(2 * (q + 0.5) * (p + 0.5) * (p + q + r + 1.5)) * 2;
      }
    }
  }
}
//-----------------------------------------------------------------------------
void tabulate_polyset_quad_derivs(xt::xtensor<double, 3>& P, std::size_t n,
                                  std::size_t nderiv,
                                  const xt::xtensor<double, 2>& x)
{
  assert(x.shape(1) == 2);
  const std::size_t m = (n + 1) * (n + 1);
  const std::size_t md = (nderiv + 1) * (nderiv + 2) / 2;

  // Indexing for quadrilateral basis functions
  auto quad_idx = [n](std::size_t px, std::size_t py) -> std::size_t
  { return (n + 1) * px + py; };

  // Compute 1D basis
  const xt::xtensor<double, 1> x0 = xt::col(x, 0);
  const xt::xtensor<double, 1> x1 = xt::col(x, 1);

  assert(x0.shape(0) > 0);
  assert(x1.shape(0) > 0);
  const auto X0 = x0 * 2.0 - 1.0;
  const auto X1 = x1 * 2.0 - 1.0;

  assert(P.shape(0) == md);
  assert(P.shape(1) == x.shape(0));
  assert(P.shape(2) == m);

  // Compute tabulation of interval for px = 0
  for (std::size_t py = 0; py <= n; ++py)
    xt::view(P, xt::all(), xt::all(), quad_idx(0, py)) = 0.0;
  xt::view(P, idx(0, 0), xt::all(), quad_idx(0, 0)) = 1.0;

  for (std::size_t ky = 0; ky <= nderiv; ++ky)
  {
    // Get reference to this derivative
    auto result = xt::view(P, idx(0, ky), xt::all(), xt::all());
    auto result0 = xt::view(P, idx(0, ky - 1), xt::all(), xt::all());
    for (std::size_t py = 1; py <= n; ++py)
    {
      const double a = 1.0 - 1.0 / static_cast<double>(py);
      xt::col(result, quad_idx(0, py))
          = X1 * xt::col(result, quad_idx(0, py - 1)) * (a + 1.0);
      if (ky > 0)
      {
        xt::col(result, quad_idx(0, py))
            += 2 * ky * xt::col(result0, quad_idx(0, py - 1)) * (a + 1.0);
      }
      if (py > 1)
      {
        xt::col(result, quad_idx(0, py))
            -= xt::col(result, quad_idx(0, py - 2)) * a;
      }
    }
  }

  // Take tensor product with another interval
  for (std::size_t kx = 0; kx <= nderiv; ++kx)
  {
    for (std::size_t px = 1; px <= n; ++px)
    {
      const double a = 1.0 - 1.0 / static_cast<double>(px);
      for (std::size_t ky = 0; ky <= nderiv - kx; ++ky)
      {
        auto result = xt::view(P, idx(kx, ky), xt::all(), xt::all());
        auto result0 = xt::view(P, idx(kx - 1, ky), xt::all(), xt::all());
        for (std::size_t py = 0; py <= n; ++py)
        {
          xt::col(result, quad_idx(px, py))
              = X0 * xt::col(result, quad_idx(px - 1, py)) * (a + 1.0);
          if (kx > 0)
          {
            xt::col(result, quad_idx(px, py))
                += 2 * kx * xt::col(result0, quad_idx(px - 1, py)) * (a + 1.0);
          }
          if (px > 1)
          {
            xt::col(result, quad_idx(px, py))
                -= xt::col(result, quad_idx(px - 2, py)) * a;
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
      xt::view(P, xt::all(), xt::all(), quad_idx(px, py))
          *= std::sqrt((2 * px + 1) * (2 * py + 1));
    }
  }
}
//-----------------------------------------------------------------------------
void tabulate_polyset_hex_derivs(xt::xtensor<double, 3>& P, std::size_t n,
                                 std::size_t nderiv,
                                 const xt::xtensor<double, 2>& x)
{
  assert(x.shape(1) == 3);
  const std::size_t m = (n + 1) * (n + 1) * (n + 1);
  const std::size_t md = (nderiv + 1) * (nderiv + 2) * (nderiv + 3) / 6;

  // Indexing for hexahedral basis functions
  auto hex_idx
      = [n](std::size_t px, std::size_t py, std::size_t pz) -> std::size_t
  { return (n + 1) * (n + 1) * px + (n + 1) * py + pz; };

  // Compute 1D basis
  const xt::xtensor<double, 1> x0 = xt::col(x, 0);
  const xt::xtensor<double, 1> x1 = xt::col(x, 1);
  const xt::xtensor<double, 1> x2 = xt::col(x, 2);

  assert(x0.shape(0) > 0);
  assert(x1.shape(0) > 0);
  assert(x2.shape(0) > 0);
  const auto X0 = x0 * 2.0 - 1.0;
  const auto X1 = x1 * 2.0 - 1.0;
  const auto X2 = x2 * 2.0 - 1.0;

  assert(P.shape(0) == md);
  assert(P.shape(1) == x.shape(0));
  assert(P.shape(2) == m);

  for (std::size_t py = 0; py <= n; ++py)
    for (std::size_t pz = 0; pz <= n; ++pz)
      xt::view(P, xt::all(), xt::all(), hex_idx(0, py, pz)) = 0.0;

  // Tabulate interval for px=py=0
  for (std::size_t kz = 0; kz <= nderiv; ++kz)
  {
    // Get reference to this derivative
    auto result = xt::view(P, idx(0, 0, kz), xt::all(), xt::all());
    if (kz == 0)
      xt::col(result, hex_idx(0, 0, 0)) = 1.0;
    else
      xt::col(result, hex_idx(0, 0, 0)) = 0.0;

    auto result0 = xt::view(P, idx(0, 0, kz - 1), xt::all(), xt::all());
    for (std::size_t pz = 1; pz <= n; ++pz)
    {
      const double a = 1.0 - 1.0 / static_cast<double>(pz);
      xt::col(result, hex_idx(0, 0, pz))
          = X2 * xt::col(result, hex_idx(0, 0, pz - 1)) * (a + 1.0);
      if (kz > 0)
      {
        xt::col(result, hex_idx(0, 0, pz))
            += 2 * kz * xt::col(result0, hex_idx(0, 0, pz - 1)) * (a + 1.0);
      }
      if (pz > 1)
      {
        xt::col(result, hex_idx(0, 0, pz))
            -= xt::col(result, hex_idx(0, 0, pz - 2)) * a;
      }
    }
  }

  // Take tensor product with interval to get quad for px=0
  for (std::size_t ky = 0; ky <= nderiv; ++ky)
  {
    for (std::size_t py = 1; py <= n; ++py)
    {
      const double a = 1.0 - 1.0 / static_cast<double>(py);
      for (std::size_t kz = 0; kz <= nderiv - ky; ++kz)
      {
        auto result = xt::view(P, idx(0, ky, kz), xt::all(), xt::all());
        auto result0 = xt::view(P, idx(0, ky - 1, kz), xt::all(), xt::all());
        for (std::size_t pz = 0; pz <= n; ++pz)
        {
          xt::col(result, hex_idx(0, py, pz))
              = X1 * xt::col(result, hex_idx(0, py - 1, pz)) * (a + 1.0);
          if (ky > 0)
          {
            xt::col(result, hex_idx(0, py, pz))
                += 2 * ky * xt::col(result0, hex_idx(0, py - 1, pz))
                   * (a + 1.0);
          }
          if (py > 1)
          {
            xt::col(result, hex_idx(0, py, pz))
                -= xt::col(result, hex_idx(0, py - 2, pz)) * a;
          }
        }
      }
    }
  }

  // Take tensor product with interval to get hex
  for (std::size_t kx = 0; kx <= nderiv; ++kx)
  {
    for (std::size_t px = 1; px <= n; ++px)
    {
      const double a = 1.0 - 1.0 / static_cast<double>(px);
      for (std::size_t ky = 0; ky <= nderiv - kx; ++ky)
      {
        for (std::size_t kz = 0; kz <= nderiv - kx - ky; ++kz)
        {
          auto result = xt::view(P, idx(kx, ky, kz), xt::all(), xt::all());
          auto result0 = xt::view(P, idx(kx - 1, ky, kz), xt::all(), xt::all());
          for (std::size_t py = 0; py <= n; ++py)
          {
            for (std::size_t pz = 0; pz <= n; ++pz)
            {
              xt::col(result, hex_idx(px, py, pz))
                  = X0 * xt::col(result, hex_idx(px - 1, py, pz)) * (a + 1.0);
              if (kx > 0)
              {
                xt::col(result, hex_idx(px, py, pz))
                    += 2 * kx * xt::col(result0, hex_idx(px - 1, py, pz))
                       * (a + 1.0);
              }
              if (px > 1)
              {
                xt::col(result, hex_idx(px, py, pz))
                    -= xt::col(result, hex_idx(px - 2, py, pz)) * a;
              }
            }
          }
        }
      }
    }
  }

  // Normalise
  for (std::size_t px = 0; px <= n; ++px)
    for (std::size_t py = 0; py <= n; ++py)
      for (std::size_t pz = 0; pz <= n; ++pz)
      {
        xt::view(P, xt::all(), xt::all(), hex_idx(px, py, pz))
            *= std::sqrt((2 * px + 1) * (2 * py + 1) * (2 * pz + 1));
      }
}
//-----------------------------------------------------------------------------
void tabulate_polyset_prism_derivs(xt::xtensor<double, 3>& P, std::size_t n,
                                   std::size_t nderiv,
                                   const xt::xtensor<double, 2>& x)
{
  assert(x.shape(1) == 3);
  const std::size_t m = (n + 1) * (n + 1) * (n + 2) / 2;
  const std::size_t md = (nderiv + 1) * (nderiv + 2) * (nderiv + 3) / 6;

  assert(P.shape(0) == md);
  assert(P.shape(1) == x.shape(0));
  assert(P.shape(2) == m);

  const auto X0 = xt::col(x, 0) * 2.0 - 1.0;
  const auto X1 = xt::col(x, 1) * 2.0 - 1.0;
  const auto X2 = xt::col(x, 2) * 2.0 - 1.0;

  assert(X0.shape(0) > 0);
  assert(X1.shape(0) > 0);
  assert(X2.shape(0) > 0);

  // Indexing for hexahedral basis functions
  auto prism_idx
      = [n](std::size_t px, std::size_t py, std::size_t pz) -> std::size_t
  {
    return (n + 1) * (py * n + 1 - (py - 2) * (py - 1) / 2) + (n + 1) * px + pz;
  };

  // f3 = ((1 - y) / 2)^2
  const auto f3 = xt::square(1.0 - X1) * 0.25;

  // Tabulate triangle for px=0
  for (std::size_t p = 0; p <= n; ++p)
    for (std::size_t q = 0; q <= n - p; ++q)
      xt::view(P, xt::all(), xt::all(), prism_idx(p, q, 0)) = 0.0;
  for (std::size_t kx = 0; kx <= nderiv; ++kx)
  {
    for (std::size_t ky = 0; ky <= nderiv - kx; ++ky)
    {
      if (kx == 0 and ky == 0)
        xt::view(P, idx(kx, ky, 0), xt::all(), prism_idx(0, 0, 0)) = 1.0;
      else
        xt::view(P, idx(kx, ky, 0), xt::all(), prism_idx(0, 0, 0)) = 0.0;

      for (std::size_t p = 1; p <= n; ++p)
      {
        auto p0 = xt::view(P, idx(kx, ky, 0), xt::all(), prism_idx(p, 0, 0));

        const double a
            = static_cast<double>(2 * p - 1) / static_cast<double>(p);
        p0 = (X0 + 0.5 * X1 + 0.5)
             * xt::view(P, idx(kx, ky, 0), xt::all(), prism_idx(p - 1, 0, 0))
             * a;
        if (kx > 0)
        {
          auto result0 = xt::view(P, idx(kx - 1, ky, 0), xt::all(),
                                  prism_idx(p - 1, 0, 0));
          p0 += 2 * kx * a * result0;
        }

        if (ky > 0)
        {
          auto result0 = xt::view(P, idx(kx, ky - 1, 0), xt::all(),
                                  prism_idx(p - 1, 0, 0));
          p0 += ky * a * result0;
        }

        if (p > 1)
        {
          // y^2 terms
          p0 -= f3
                * xt::view(P, idx(kx, ky, 0), xt::all(), prism_idx(p - 2, 0, 0))
                * (a - 1.0);
          if (ky > 0)
          {
            auto result0 = xt::view(P, idx(kx, ky - 1, 0), xt::all(),
                                    prism_idx(p - 2, 0, 0));
            p0 -= ky * (X1 - 1.0) * result0 * (a - 1.0);
          }

          if (ky > 1)
          {
            auto result0 = xt::view(P, idx(kx, ky - 2, 0), xt::all(),
                                    prism_idx(p - 2, 0, 0));
            p0 -= ky * (ky - 1) * result0 * (a - 1.0);
          }
        }
      }

      for (std::size_t p = 0; p < n; ++p)
      {
        auto p0 = xt::view(P, idx(kx, ky, 0), xt::all(), prism_idx(p, 0, 0));
        auto p1 = xt::view(P, idx(kx, ky, 0), xt::all(), prism_idx(p, 1, 0));
        p1 = p0 * (X1 * (1.5 + p) + 0.5 + p);
        if (ky > 0)
        {
          auto result0
              = xt::view(P, idx(kx, ky - 1, 0), xt::all(), prism_idx(p, 0, 0));
          p1 += 2 * ky * (1.5 + p) * result0;
        }

        for (std::size_t q = 1; q < n - p; ++q)
        {
          const auto [a1, a2, a3] = jrc(2 * p + 1, q);
          xt::view(P, idx(kx, ky, 0), xt::all(), prism_idx(p, q + 1, 0))
              = xt::view(P, idx(kx, ky, 0), xt::all(), prism_idx(p, q, 0))
                    * (X1 * a1 + a2)
                - xt::view(P, idx(kx, ky, 0), xt::all(), prism_idx(p, q - 1, 0))
                      * a3;
          if (ky > 0)
          {
            auto result0 = xt::view(P, idx(kx, ky - 1, 0), xt::all(),
                                    prism_idx(p, q, 0));
            xt::view(P, idx(kx, ky, 0), xt::all(), prism_idx(p, q + 1, 0))
                += 2 * ky * a1 * result0;
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
      const double a = 1.0 - 1.0 / static_cast<double>(r);
      for (std::size_t kx = 0; kx <= nderiv - kz; ++kx)
      {
        for (std::size_t ky = 0; ky <= nderiv - kx - kz; ++ky)
        {
          auto result = xt::view(P, idx(kx, ky, kz), xt::all(), xt::all());
          auto result0 = xt::view(P, idx(kx, ky, kz - 1), xt::all(), xt::all());
          for (std::size_t p = 0; p <= n; ++p)
          {
            for (std::size_t q = 0; q <= n - p; ++q)
            {
              xt::col(result, prism_idx(p, q, r))
                  = X2 * xt::col(result, prism_idx(p, q, r - 1)) * (a + 1.0);
              if (kz > 0)
                xt::col(result, prism_idx(p, q, r))
                    += 2 * kz * xt::col(result0, prism_idx(p, q, r - 1))
                       * (a + 1.0);
              if (r > 1)
                xt::col(result, prism_idx(p, q, r))
                    -= xt::col(result, prism_idx(p, q, r - 2)) * a;
            }
          }
        }
      }
    }
  }

  // Normalise
  for (std::size_t p = 0; p <= n; ++p)
    for (std::size_t q = 0; q <= n - p; ++q)
      for (std::size_t r = 0; r <= n; ++r)
        xt::view(P, xt::all(), xt::all(), prism_idx(p, q, r))
            *= std::sqrt((p + 0.5) * (p + q + 1) * (2 * r + 1)) * 2;
}
} // namespace
//-----------------------------------------------------------------------------
void polyset::tabulate(xt::xtensor<double, 3>& P, cell::type celltype, int d,
                       int n, const xt::xarray<double>& x)
{
  switch (celltype)
  {
  case cell::type::point:
    return tabulate_polyset_point_derivs(P, d, n, x);
  case cell::type::interval:
    assert(x.dimension() == 1);
    return tabulate_polyset_line_derivs(P, d, n, x);
  case cell::type::triangle:
    return tabulate_polyset_triangle_derivs(P, d, n, x);
  case cell::type::tetrahedron:
    return tabulate_polyset_tetrahedron_derivs(P, d, n, x);
  case cell::type::quadrilateral:
    return tabulate_polyset_quad_derivs(P, d, n, x);
  case cell::type::prism:
    return tabulate_polyset_prism_derivs(P, d, n, x);
  case cell::type::pyramid:
    return tabulate_polyset_pyramid_derivs(P, d, n, x);
  case cell::type::hexahedron:
    return tabulate_polyset_hex_derivs(P, d, n, x);
  default:
    throw std::runtime_error("Polynomial set: unsupported cell type");
  }
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 3> polyset::tabulate(cell::type celltype, int d, int n,
                                         const xt::xarray<double>& x)
{
  xt::xtensor<double, 3> out(
      {static_cast<std::size_t>(polyset::nderivs(celltype, n)), x.shape(0),
       static_cast<std::size_t>(polyset::dim(celltype, d))});
  polyset::tabulate(out, celltype, d, n, x);
  return out;
}
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
