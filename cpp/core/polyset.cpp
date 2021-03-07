// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "polyset.h"
#include "cell.h"
#include "indexing.h"
#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

using namespace basix;

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
// Compute the complete set of derivatives from 0 to nderiv, for all the
// polynomials up to order n on a line segment. The polynomials used are
// Legendre Polynomials, with the recurrence relation given by
// n P(n) = (2n - 1) x P_{n-1} - (n - 1) P_{n-2} in the interval [-1, 1]. The
// range is rescaled here to [0, 1].
xt::xtensor<double, 3>
tabulate_polyset_line_derivs(std::size_t degree, std::size_t nderiv,
                             const xt::xtensor<double, 1>& x)
{
  assert(x.shape()[0] > 0);
  const auto X = x * 2.0 - 1.0;
  const std::size_t m = (degree + 1);
  xt::xtensor<double, 3> dresult({nderiv + 1, x.shape()[0], m});
  for (std::size_t k = 0; k <= nderiv; ++k)
  {
    // Get reference to this derivative
    auto result = xt::view(dresult, k, xt::all(), xt::all());
    if (k == 0)
      xt::col(result, 0) = 1.0;
    else
      xt::col(result, 0) = 0.0;

    auto result0 = xt::view(dresult, k - 1, xt::all(), xt::all());
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
  for (std::size_t k = 0; k < nderiv + 1; ++k)
    for (std::size_t p = 0; p <= degree; ++p)
      xt::view(dresult, k, xt::all(), p) *= std::sqrt(p + 0.5);

  return dresult;
}
//-----------------------------------------------------------------------------
// Compute the complete set of derivatives from 0 to nderiv, for all the
// polynomials up to order n on a triangle in [0, 1][0, 1]. The
// polynomials P_{pq} are built up in sequence, firstly along q = 0,
// which is a line segment, as in tabulate_polyset_interval_derivs
// above, but with a change of variables. The polynomials are then
// extended in the q direction, using the relation given in Sherwin and
// Karniadakis 1995 (https://doi.org/10.1016/0045-7825(94)00745-9).
template <typename Mat>
xt::xtensor<double, 3> tabulate_polyset_triangle_derivs(int n, int nderiv,
                                                        const Mat& pts)
// xt::xtensor<double, 3>
// tabulate_polyset_triangle_derivs(int n, int nderiv,
//                                  const xt::xtensor<double, 2>& pts)
{
  // std::cout << "Tri 0" << std::endl;
  assert(pts.shape()[1] == 2);

  const auto x = pts * 2.0 - 1.0;
  auto x0 = xt::col(x, 0);
  auto x1 = xt::col(x, 1);

  const std::size_t m = (n + 1) * (n + 2) / 2;
  const std::size_t md = (nderiv + 1) * (nderiv + 2) / 2;
  xt::xtensor<double, 3> dresult({md, pts.shape()[0], m});

  // f3 = ((1 - y) / 2)^2
  const auto f3 = xt::square(1.0 - x1) * 0.25;

  // Iterate over derivatives in increasing order, since higher derivatives

  // Depend on earlier calculations
  xt::xtensor<double, 2> result({pts.shape()[0], m});
  for (int k = 0; k <= nderiv; ++k)
  {
    for (int kx = 0; kx <= k; ++kx)
    {
      const int ky = k - kx;
      if (kx == 0 and ky == 0)
        xt::col(result, 0) = 1.0;
      else
        xt::col(result, 0) = 0.0;

      for (int p = 1; p < n + 1; ++p)
      {
        auto p0 = xt::col(result, idx(p, 0));
        const double a
            = static_cast<double>(2 * p - 1) / static_cast<double>(p);
        p0 = (x0 + 0.5 * x1 + 0.5) * xt::col(result, idx(p - 1, 0)) * a;
        if (kx > 0)
        {
          auto result0
              = xt::view(dresult, idx(kx - 1, ky), xt::all(), idx(p - 1, 0));
          p0 += 2 * kx * a * result0;
        }

        if (ky > 0)
        {
          auto result0
              = xt::view(dresult, idx(kx, ky - 1), xt::all(), idx(p - 1, 0));
          p0 += ky * a * result0;
        }

        if (p > 1)
        {
          // y^2 terms
          p0 -= f3 * xt::col(result, idx(p - 2, 0)) * (a - 1.0);
          if (ky > 0)
          {
            auto result0
                = xt::view(dresult, idx(kx, ky - 1), xt::all(), idx(p - 2, 0));
            p0 -= ky * (x1 - 1.0) * result0 * (a - 1.0);
          }

          if (ky > 1)
          {
            auto result0
                = xt::view(dresult, idx(kx, ky - 2), xt::all(), idx(p - 2, 0));
            p0 -= ky * (ky - 1) * result0 * (a - 1.0);
          }
        }
      }

      for (int p = 0; p < n; ++p)
      {
        auto p0 = xt::col(result, idx(p, 0));
        auto p1 = xt::col(result, idx(p, 1));
        p1 = p0 * (x1 * (1.5 + p) + 0.5 + p);
        if (ky > 0)
        {
          auto result0
              = xt::view(dresult, idx(kx, ky - 1), xt::all(), idx(p, 0));
          p1 += 2 * ky * (1.5 + p) * result0;
        }

        for (int q = 1; q < n - p; ++q)
        {
          const auto [a1, a2, a3] = jrc(2 * p + 1, q);
          xt::col(result, idx(p, q + 1))
              = xt::col(result, idx(p, q)) * (x1 * a1 + a2)
                - xt::col(result, idx(p, q - 1)) * a3;
          if (ky > 0)
          {
            auto result0
                = xt::view(dresult, idx(kx, ky - 1), xt::all(), idx(p, q));
            xt::col(result, idx(p, q + 1)) += 2 * ky * a1 * result0;
          }
        }
      }

      // Store
      xt::view(dresult, idx(kx, ky), xt::all(), xt::all()) = result;
    }
  }

  // Normalisation
  for (std::size_t j = 0; j < dresult.shape()[0]; ++j)
  {
    auto result = xt::view(dresult, j, xt::all(), xt::all());
    for (int p = 0; p <= n; ++p)
      for (int q = 0; q <= n - p; ++q)
        xt::col(result, idx(p, q)) *= std::sqrt((p + 0.5) * (p + q + 1));
  }

  return dresult;
}
//-----------------------------------------------------------------------------
std::vector<Eigen::ArrayXXd>
tabulate_polyset_tetrahedron_derivs(int n, int nderiv,
                                    const Eigen::ArrayXXd& pts)
{
  assert(pts.cols() == 3);

  Eigen::ArrayXXd x = pts * 2.0 - 1.0;

  const int m = (n + 1) * (n + 2) * (n + 3) / 6;
  const int md = (nderiv + 1) * (nderiv + 2) * (nderiv + 3) / 6;
  std::vector<Eigen::ArrayXXd> dresult(md);

  const Eigen::ArrayXd f2 = (x.col(1) + x.col(2)).square() * 0.25;
  const Eigen::ArrayXd f3 = (1.0 + x.col(1) * 2.0 + x.col(2)) * 0.5;
  const Eigen::ArrayXd f4 = (1.0 - x.col(2)) * 0.5;
  const Eigen::ArrayXd f5 = f4 * f4;

  // Traverse derivatives in increasing order
  Eigen::ArrayXXd result(pts.rows(), m);
  for (int k = 0; k < nderiv + 1; ++k)
  {
    for (int j = 0; j < k + 1; ++j)
    {
      for (int kx = 0; kx < j + 1; ++kx)
      {
        const int ky = j - kx;
        const int kz = k - j;
        if (kx == 0 and ky == 0 and kz == 0)
          result.col(0).fill(1.0);
        else
          result.col(0).setZero();

        for (int p = 1; p < n + 1; ++p)
        {
          double a = static_cast<double>(2 * p - 1) / static_cast<double>(p);
          result.col(idx(p, 0, 0))
              = (x.col(0) + 0.5 * (x.col(1) + x.col(2)) + 1.0)
                * result.col(idx(p - 1, 0, 0)) * a;
          if (kx > 0)
          {
            result.col(idx(p, 0, 0))
                += 2 * kx * a
                   * dresult[idx(kx - 1, ky, kz)].col(idx(p - 1, 0, 0));
          }

          if (ky > 0)
          {
            result.col(idx(p, 0, 0))
                += ky * a * dresult[idx(kx, ky - 1, kz)].col(idx(p - 1, 0, 0));
          }

          if (kz > 0)
          {
            result.col(idx(p, 0, 0))
                += kz * a * dresult[idx(kx, ky, kz - 1)].col(idx(p - 1, 0, 0));
          }

          if (p > 1)
          {
            result.col(idx(p, 0, 0))
                -= f2 * result.col(idx(p - 2, 0, 0)) * (a - 1.0);
            if (ky > 0)
            {
              result.col(idx(p, 0, 0))
                  -= ky * (x.col(1) + x.col(2))
                     * dresult[idx(kx, ky - 1, kz)].col(idx(p - 2, 0, 0))
                     * (a - 1.0);
            }

            if (ky > 1)
            {
              result.col(idx(p, 0, 0))
                  -= ky * (ky - 1)
                     * dresult[idx(kx, ky - 2, kz)].col(idx(p - 2, 0, 0))
                     * (a - 1.0);
            }

            if (kz > 0)
            {
              result.col(idx(p, 0, 0))
                  -= kz * (x.col(1) + x.col(2))
                     * dresult[idx(kx, ky, kz - 1)].col(idx(p - 2, 0, 0))
                     * (a - 1.0);
            }

            if (kz > 1)
            {
              result.col(idx(p, 0, 0))
                  -= kz * (kz - 1)
                     * dresult[idx(kx, ky, kz - 2)].col(idx(p - 2, 0, 0))
                     * (a - 1.0);
            }

            if (ky > 0 and kz > 0)
            {
              result.col(idx(p, 0, 0))
                  -= 2.0 * ky * kz
                     * dresult[idx(kx, ky - 1, kz - 1)].col(idx(p - 2, 0, 0))
                     * (a - 1.0);
            }
          }
        }

        for (int p = 0; p < n; ++p)
        {
          result.col(idx(p, 1, 0))
              = result.col(idx(p, 0, 0))
                * ((1.0 + x.col(1)) * p
                   + (2.0 + x.col(1) * 3.0 + x.col(2)) * 0.5);
          if (ky > 0)
          {
            result.col(idx(p, 1, 0))
                += 2 * ky * dresult[idx(kx, ky - 1, kz)].col(idx(p, 0, 0))
                   * (1.5 + p);
          }

          if (kz > 0)
          {
            result.col(idx(p, 1, 0))
                += kz * dresult[idx(kx, ky, kz - 1)].col(idx(p, 0, 0));
          }

          for (int q = 1; q < n - p; ++q)
          {
            auto [aq, bq, cq] = jrc(2 * p + 1, q);
            result.col(idx(p, q + 1, 0))
                = result.col(idx(p, q, 0)) * (f3 * aq + f4 * bq)
                  - result.col(idx(p, q - 1, 0)) * f5 * cq;

            if (ky > 0)
            {
              result.col(idx(p, q + 1, 0))
                  += 2 * ky * dresult[idx(kx, ky - 1, kz)].col(idx(p, q, 0))
                     * aq;
            }

            if (kz > 0)
            {
              result.col(idx(p, q + 1, 0))
                  += kz * dresult[idx(kx, ky, kz - 1)].col(idx(p, q, 0))
                         * (aq - bq)
                     + kz * (1.0 - x.col(2))
                           * dresult[idx(kx, ky, kz - 1)].col(idx(p, q - 1, 0))
                           * cq;
            }

            if (kz > 1)
            {
              // Quadratic term in z
              result.col(idx(p, q + 1, 0))
                  -= kz * (kz - 1)
                     * dresult[idx(kx, ky, kz - 2)].col(idx(p, q - 1, 0)) * cq;
            }
          }
        }

        for (int p = 0; p < n; ++p)
        {
          for (int q = 0; q < n - p; ++q)
          {
            result.col(idx(p, q, 1))
                = result.col(idx(p, q, 0))
                  * ((1.0 + p + q) + x.col(2) * (2.0 + p + q));
            if (kz > 0)
            {
              result.col(idx(p, q, 1))
                  += 2 * kz * (2.0 + p + q)
                     * dresult[idx(kx, ky, kz - 1)].col(idx(p, q, 0));
            }
          }
        }

        for (int p = 0; p < n - 1; ++p)
        {
          for (int q = 0; q < n - p - 1; ++q)
          {
            for (int r = 1; r < n - p - q; ++r)
            {
              auto [ar, br, cr] = jrc(2 * p + 2 * q + 2, r);
              result.col(idx(p, q, r + 1))
                  = result.col(idx(p, q, r)) * (x.col(2) * ar + br)
                    - result.col(idx(p, q, r - 1)) * cr;
              if (kz > 0)
              {
                result.col(idx(p, q, r + 1))
                    += 2 * kz * ar
                       * dresult[idx(kx, ky, kz - 1)].col(idx(p, q, r));
              }
            }
          }
        }

        // Store this derivative
        dresult[idx(kx, ky, kz)] = result;
      }
    }
  }

  for (Eigen::ArrayXXd& result : dresult)
  {
    for (int p = 0; p < n + 1; ++p)
    {
      for (int q = 0; q < n - p + 1; ++q)
      {
        for (int r = 0; r < n - p - q + 1; ++r)
        {
          result.col(idx(p, q, r))
              *= std::sqrt((p + 0.5) * (p + q + 1.0) * (p + q + r + 1.5));
        }
      }
    }
  }

  return dresult;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 3>
tabulate_polyset_pyramid_derivs(std::size_t n, std::size_t nderiv,
                                const xt::xtensor<double, 2>& pts)
{
  assert(pts.shape()[1] == 3);
  const std::size_t m = (n + 1) * (n + 2) * (2 * n + 3) / 6;
  const std::size_t md = (nderiv + 1) * (nderiv + 2) * (nderiv + 3) / 6;

  // Indexing for pyramidal basis functions
  auto pyr_idx = [n](int p, int q, int r) -> std::size_t {
    const int rv = n - r + 1;
    const int r0 = r * (n + 1) * (n - r + 2) + (2 * r - 1) * (r - 1) * r / 6;
    return r0 + p * rv + q;
  };

  auto x = pts * 2.0 - 1.0;
  const auto x0 = xt::col(x, 0);
  const auto x1 = xt::col(x, 1);
  const auto x2 = xt::col(x, 2);

  auto f2 = 0.25 * xt::square(1.0 - x2);

  // Traverse derivatives in increasing order
  xt::xtensor<double, 3> dresult({md, pts.shape()[0], m});
  xt::xtensor<double, 2> result({pts.shape()[0], m});
  for (std::size_t k = 0; k < nderiv + 1; ++k)
  {
    for (std::size_t j = 0; j < k + 1; ++j)
    {
      for (std::size_t kx = 0; kx < j + 1; ++kx)
      {
        result = xt::zeros<double>(result.shape());
        const std::size_t ky = j - kx;
        const std::size_t kz = k - j;

        const std::size_t pyramidal_index = pyr_idx(0, 0, 0);
        assert(pyramidal_index < m);
        if (kx == 0 and ky == 0 and kz == 0)
          xt::col(result, pyramidal_index) = 1.0;
        else
          xt::col(result, pyramidal_index) = 0.0;

        // r = 0
        for (std::size_t p = 0; p < n + 1; ++p)
        {
          if (p > 0)
          {
            const double a
                = static_cast<double>(p - 1) / static_cast<double>(p);
            auto p00 = xt::col(result, pyr_idx(p, 0, 0));
            p00 = (0.5 + x0 + x2 * 0.5) * xt::col(result, pyr_idx(p - 1, 0, 0))
                  * (a + 1.0);

            if (kx > 0)
            {
              p00 += 2.0 * kx
                     * xt::view(dresult, idx(kx - 1, ky, kz), xt::all(),
                                pyr_idx(p - 1, 0, 0))
                     * (a + 1.0);
            }

            if (kz > 0)
            {
              p00 += kz
                     * xt::view(dresult, idx(kx, ky, kz - 1), xt::all(),
                                pyr_idx(p - 1, 0, 0))
                     * (a + 1.0);
            }

            if (p > 1)
            {
              p00 -= f2 * xt::col(result, pyr_idx(p - 2, 0, 0)) * a;

              if (kz > 0)
              {
                p00 += kz * (1.0 - x2)
                       * xt::view(dresult, idx(kx, ky, kz - 1), xt::all(),
                                  pyr_idx(p - 2, 0, 0))
                       * a;
              }

              if (kz > 1)
              {
                // quadratic term in z
                p00 -= kz * (kz - 1)
                       * xt::view(dresult, idx(kx, ky, kz - 2), xt::all(),
                                  pyr_idx(p - 2, 0, 0))
                       * a;
              }
            }
          }

          for (std::size_t q = 1; q < n + 1; ++q)
          {
            const double a
                = static_cast<double>(q - 1) / static_cast<double>(q);
            auto r_pq = xt::col(result, pyr_idx(p, q, 0));
            r_pq = (0.5 + x1 + x2 * 0.5) * xt::col(result, pyr_idx(p, q - 1, 0))
                   * (a + 1.0);
            if (ky > 0)
            {
              r_pq += 2.0 * ky
                      * xt::view(dresult, idx(kx, ky - 1, kz), xt::all(),
                                 pyr_idx(p, q - 1, 0))
                      * (a + 1.0);
            }

            if (kz > 0)
            {
              r_pq += kz
                      * xt::view(dresult, idx(kx, ky, kz - 1), xt::all(),
                                 pyr_idx(p, q - 1, 0))
                      * (a + 1.0);
            }

            if (q > 1)
            {
              r_pq -= f2 * xt::col(result, pyr_idx(p, q - 2, 0)) * a;

              if (kz > 0)
              {
                r_pq += kz * (1.0 - x2)
                        * xt::view(dresult, idx(kx, ky, kz - 1), xt::all(),
                                   pyr_idx(p, q - 2, 0))
                        * a;
              }

              if (kz > 1)
              {
                r_pq -= kz * (kz - 1)
                        * xt::view(dresult, idx(kx, ky, kz - 2),
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
            auto r_pq1 = xt::col(result, pyr_idx(p, q, 1));
            r_pq1 = xt::col(result, pyr_idx(p, q, 0))
                    * ((1.0 + p + q) + x2 * (2.0 + p + q));
            if (kz > 0)
            {
              r_pq1 += 2 * kz
                       * xt::view(dresult, idx(kx, ky, kz - 1), xt::all(),
                                  pyr_idx(p, q, 0))
                       * (2.0 + p + q);
            }
          }
        }

        for (std::size_t r = 1; r < n + 1; ++r)
        {
          for (std::size_t p = 0; p < n - r; ++p)
          {
            for (std::size_t q = 0; q < n - r; ++q)
            {
              auto [ar, br, cr] = jrc(2 * p + 2 * q + 2, r);
              auto r_pqr = xt::col(result, pyr_idx(p, q, r + 1));
              r_pqr = xt::col(result, pyr_idx(p, q, r)) * (x2 * ar + br)
                      - xt::col(result, pyr_idx(p, q, r - 1)) * cr;
              if (kz > 0)
              {
                r_pqr += ar * 2 * kz
                         * xt::view(dresult, idx(kx, ky, kz - 1), xt::all(),
                                    pyr_idx(p, q, r));
              }
            }
          }
        }

        xt::view(dresult, idx(kx, ky, kz), xt::all(), xt::all()) = result;
      }
    }
  }

  for (std::size_t i = 0; i < dresult.shape()[0]; ++i)
  {
    auto result = xt::view(dresult, i, xt::all(), xt::all());
    for (std::size_t r = 0; r < n + 1; ++r)
    {
      for (std::size_t p = 0; p < n - r + 1; ++p)
      {
        for (std::size_t q = 0; q < n - r + 1; ++q)
        {
          xt::col(result, pyr_idx(p, q, r))
              *= std::sqrt((q + 0.5) * (p + 0.5) * (p + q + r + 1.5));
        }
      }
    }
  }

  return dresult;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 3>
tabulate_polyset_quad_derivs(int n, int nderiv,
                             const xt::xtensor<double, 2>& pts)
{
  assert(pts.shape()[1] == 2);
  const std::size_t m = (n + 1) * (n + 1);
  const std::size_t md = (nderiv + 1) * (nderiv + 2) / 2;

  // Compute 1D basis
  const xt::xtensor<double, 1> x0 = xt::col(pts, 0);
  const xt::xtensor<double, 1> x1 = xt::col(pts, 1);
  xt::xtensor<double, 3> px = tabulate_polyset_line_derivs(n, nderiv, x0);
  xt::xtensor<double, 3> py = tabulate_polyset_line_derivs(n, nderiv, x1);

  xt::xtensor<double, 3> dresult({md, pts.shape()[0], m});
  for (int kx = 0; kx < nderiv + 1; ++kx)
  {
    auto p0 = xt::view(px, kx, xt::all(), xt::all());
    for (int ky = 0; ky < nderiv + 1 - kx; ++ky)
    {
      auto result = xt::view(dresult, idx(kx, ky), xt::all(), xt::all());
      auto p1 = xt::view(py, ky, xt::all(), xt::all());
      int c = 0;
      for (std::size_t i = 0; i < p0.shape()[1]; ++i)
        for (std::size_t j = 0; j < p0.shape()[1]; ++j)
          xt::col(result, c++) = xt::col(p0, i) * xt::col(p1, j);
    }
  }

  return dresult;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 3>
tabulate_polyset_hex_derivs(std::size_t n, std::size_t nderiv,
                            const xt::xtensor<double, 2>& x)
{
  assert(x.shape()[1] == 3);
  const std::size_t m = (n + 1) * (n + 1) * (n + 1);
  const std::size_t md = (nderiv + 1) * (nderiv + 2) * (nderiv + 3) / 6;

  // Compute 1D basis
  const xt::xtensor<double, 1> x0 = xt::col(x, 0);
  const xt::xtensor<double, 1> x1 = xt::col(x, 1);
  const xt::xtensor<double, 1> x2 = xt::col(x, 2);
  xt::xtensor<double, 3> px = tabulate_polyset_line_derivs(n, nderiv, x0);
  xt::xtensor<double, 3> py = tabulate_polyset_line_derivs(n, nderiv, x1);
  xt::xtensor<double, 3> pz = tabulate_polyset_line_derivs(n, nderiv, x2);

  // Compute basis
  xt::xtensor<double, 3> dresult({md, x.shape()[0], m});
  for (std::size_t kx = 0; kx < nderiv + 1; ++kx)
  {
    auto p0 = xt::view(px, kx, xt::all(), xt::all());
    for (std::size_t ky = 0; ky < nderiv + 1 - kx; ++ky)
    {
      auto p1 = xt::view(py, ky, xt::all(), xt::all());
      for (std::size_t kz = 0; kz < nderiv + 1 - kx - ky; ++kz)
      {
        auto result = xt::view(dresult, idx(kx, ky, kz), xt::all(), xt::all());
        auto p2 = xt::view(pz, kz, xt::all(), xt::all());
        int c = 0;
        for (std::size_t i = 0; i < p0.shape()[1]; ++i)
        {
          auto pi = xt::col(p0, i);
          for (std::size_t j = 0; j < p1.shape()[1]; ++j)
          {
            auto pj = xt::col(p1, j);
            for (std::size_t k = 0; k < p2.shape()[1]; ++k)
              xt::col(result, c++) = pi * pj * xt::col(p2, k);
          }
        }
      }
    }
  }

  return dresult;
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 3>
tabulate_polyset_prism_derivs(std::size_t n, std::size_t nderiv,
                              const xt::xtensor<double, 2>& x)
{
  assert(x.shape()[1] == 3);
  const std::size_t m = (n + 1) * (n + 1) * (n + 2) / 2;
  const std::size_t md = (nderiv + 1) * (nderiv + 2) * (nderiv + 3) / 6;

  const xt::xtensor<double, 2> x01 = xt::view(x, xt::all(), xt::range(0, 2));
  const xt::xtensor<double, 1> x2 = xt::col(x, 2);
  xt::xtensor<double, 3> pxy = tabulate_polyset_triangle_derivs(n, nderiv, x01);
  xt::xtensor<double, 3> pz = tabulate_polyset_line_derivs(n, nderiv, x2);

  xt::xtensor<double, 3> dresult({md, x.shape()[0], m});
  for (std::size_t kx = 0; kx < nderiv + 1; ++kx)
  {
    for (std::size_t ky = 0; ky < nderiv + 1 - kx; ++ky)
    {
      auto p0 = xt::view(pxy, idx(kx, ky), xt::all(), xt::all());
      for (std::size_t kz = 0; kz < nderiv + 1 - kx - ky; ++kz)
      {
        auto p1 = xt::view(pz, kz, xt::all(), xt::all());
        auto result = xt::view(dresult, idx(kx, ky, kz), xt::all(), xt::all());
        int c = 0;
        for (std::size_t i = 0; i < p0.shape()[1]; ++i)
          for (std::size_t k = 0; k < p1.shape()[1]; ++k)
            xt::col(result, c++) = xt::col(p0, i) * xt::col(p1, k);
      }
    }
  }

  return dresult;
}
//-----------------------------------------------------------------------------
// xt::xtensor<double, 3> _tabulate(cell::type celltype, int d, int n,
//                                  const xt::xtensor<double, 2>& x)
// {
//   switch (celltype)
//   {
//   case cell::type::interval:
//     return tabulate_polyset_line_derivs(d, n, x);
//   case cell::type::triangle:
//     return tabulate_polyset_triangle_derivs(d, n, x);
//   case cell::type::tetrahedron:
//     return tabulate_polyset_tetrahedron_derivs(d, n, x);
//   case cell::type::quadrilateral:
//     return tabulate_polyset_quad_derivs(d, n, x);
//   case cell::type::prism:
//     return tabulate_polyset_prism_derivs(d, n, x);
//   case cell::type::pyramid:
//     return tabulate_polyset_pyramid_derivs(d, n, x);
//   case cell::type::hexahedron:
//     return tabulate_polyset_hex_derivs(d, n, x);
//   default:
//     throw std::runtime_error("Polynomial set: Unsupported cell type");
//   }
// }
} // namespace
//-----------------------------------------------------------------------------
std::vector<Eigen::ArrayXXd> polyset::tabulate(cell::type celltype, int d,
                                               int n, const Eigen::ArrayXXd& x)
{
  switch (celltype)
  {
  case cell::type::interval:
  {
    assert(x.cols() == 1);
    std::array<std::size_t, 1> s = {(std::size_t)x.rows()};
    auto _x = xt::adapt(x.data(), x.size(), xt::no_ownership(), s);
    auto tab = tabulate_polyset_line_derivs(d, n, _x);
    std::vector<Eigen::ArrayXXd> t;
    for (std::size_t i = 0; i < tab.shape()[0]; ++i)
    {
      Eigen::ArrayXXd mat(tab.shape()[1], tab.shape()[2]);
      for (std::size_t j = 0; j < tab.shape()[1]; ++j)
        for (std::size_t k = 0; k < tab.shape()[2]; ++k)
          mat(j, k) = tab(i, j, k);
      t.push_back(mat);
    }

    return t;
  }
  case cell::type::triangle:
  {
    std::array<std::size_t, 2> s
        = {(std::size_t)x.rows(), (std::size_t)x.cols()};
    auto _x = xt::adapt<xt::layout_type::column_major>(x.data(), x.size(),
                                                       xt::no_ownership(), s);
    auto tab = tabulate_polyset_triangle_derivs(d, n, _x);
    std::vector<Eigen::ArrayXXd> t;
    for (std::size_t i = 0; i < tab.shape()[0]; ++i)
    {
      Eigen::ArrayXXd mat(tab.shape()[1], tab.shape()[2]);
      for (std::size_t j = 0; j < tab.shape()[1]; ++j)
        for (std::size_t k = 0; k < tab.shape()[2]; ++k)
          mat(j, k) = tab(i, j, k);
      t.push_back(mat);
    }
    return t;
  }
  case cell::type::tetrahedron:
    return tabulate_polyset_tetrahedron_derivs(d, n, x);
  case cell::type::quadrilateral:
  {
    std::array<std::size_t, 2> s
        = {(std::size_t)x.rows(), (std::size_t)x.cols()};
    auto _x = xt::adapt<xt::layout_type::column_major>(x.data(), x.size(),
                                                       xt::no_ownership(), s);
    auto tab = tabulate_polyset_quad_derivs(d, n, _x);
    std::vector<Eigen::ArrayXXd> t;
    for (std::size_t i = 0; i < tab.shape()[0]; ++i)
    {
      Eigen::ArrayXXd mat(tab.shape()[1], tab.shape()[2]);
      for (std::size_t j = 0; j < tab.shape()[1]; ++j)
        for (std::size_t k = 0; k < tab.shape()[2]; ++k)
          mat(j, k) = tab(i, j, k);
      t.push_back(mat);
    }
    return t;
  }
  case cell::type::prism:
  {
    std::array<std::size_t, 2> s
        = {(std::size_t)x.rows(), (std::size_t)x.cols()};
    auto _x = xt::adapt<xt::layout_type::column_major>(x.data(), x.size(),
                                                       xt::no_ownership(), s);
    auto tab = tabulate_polyset_prism_derivs(d, n, _x);
    std::vector<Eigen::ArrayXXd> t;
    for (std::size_t i = 0; i < tab.shape()[0]; ++i)
    {
      Eigen::ArrayXXd mat(tab.shape()[1], tab.shape()[2]);
      for (std::size_t j = 0; j < tab.shape()[1]; ++j)
        for (std::size_t k = 0; k < tab.shape()[2]; ++k)
          mat(j, k) = tab(i, j, k);
      t.push_back(mat);
    }
    return t;
  }
  case cell::type::pyramid:
  {
    std::array<std::size_t, 2> s
        = {(std::size_t)x.rows(), (std::size_t)x.cols()};
    auto _x = xt::adapt<xt::layout_type::column_major>(x.data(), x.size(),
                                                       xt::no_ownership(), s);
    auto tab = tabulate_polyset_pyramid_derivs(d, n, _x);
    std::vector<Eigen::ArrayXXd> t;
    for (std::size_t i = 0; i < tab.shape()[0]; ++i)
    {
      Eigen::ArrayXXd mat(tab.shape()[1], tab.shape()[2]);
      for (std::size_t j = 0; j < tab.shape()[1]; ++j)
        for (std::size_t k = 0; k < tab.shape()[2]; ++k)
          mat(j, k) = tab(i, j, k);
      t.push_back(mat);
    }
    return t;
  }
  case cell::type::hexahedron:
  {
    std::array<std::size_t, 2> s
        = {(std::size_t)x.rows(), (std::size_t)x.cols()};
    auto _x = xt::adapt<xt::layout_type::column_major>(x.data(), x.size(),
                                                       xt::no_ownership(), s);
    auto tab = tabulate_polyset_hex_derivs(d, n, _x);
    std::vector<Eigen::ArrayXXd> t;
    for (std::size_t i = 0; i < tab.shape()[0]; ++i)
    {
      Eigen::ArrayXXd mat(tab.shape()[1], tab.shape()[2]);
      for (std::size_t j = 0; j < tab.shape()[1]; ++j)
        for (std::size_t k = 0; k < tab.shape()[2]; ++k)
          mat(j, k) = tab(i, j, k);
      t.push_back(mat);
    }
    return t;
  }
  default:
    throw std::runtime_error("Polynomial set: Unsupported cell type");
  }

  // std::array<std::size_t> s = {x.cols(), x.rows()};
  // xt::xtensor<double, 2> _x
  //     = xt::transpose(xt::adapt(x.data(), x.size(), shape));
  // auto tab = _tabulate(celltype, d, n, _x);

  // std::vector<Eigen::ArrayXXd> t;
  // for (std::size_t i = 0; i < tab.shape()[0]; ++i)
  // {
  //   Eigen::ArrayXXd mat(tab.shape()[1], tab.shape()[2]);
  //   for (std::size_t j = 0; j < tab.shape()[1]; ++j)
  //     for (std::size_t k = 0; k < tab.shape()[2]; ++k)
  //       mat(j, k) = tab(i, j, k);
  //   t.push_back(mat);
  // }
}
//-----------------------------------------------------------------------------
int polyset::dim(cell::type celltype, int d)
{
  switch (celltype)
  {
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
