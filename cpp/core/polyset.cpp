// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "polyset.h"
#include "cell.h"
#include "indexing.h"
#include <Eigen/Dense>
#include <array>
#include <cmath>

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
Eigen::ArrayXXd tabulate_polyset_line_derivs(int degree, int nderiv,
                                             const Eigen::ArrayXXd& x)
{
  assert(x.cols() == 1);

  const int npoints = x.rows();
  const int m = (degree + 1);

  Eigen::ArrayXXd result(npoints, m * (nderiv + 1));
  for (int k = 0; k < nderiv + 1; ++k)
  {
    const int base_col = k * m;

    if (k == 0)
      result.col(base_col).fill(1.0);
    else
      result.col(base_col).setZero();

    for (int p = 1; p < degree + 1; ++p)
    {
      const double a = 1.0 - 1.0 / static_cast<double>(p);
      result.col(base_col + p)
          = (x * 2.0 - 1.0) * result.col(base_col + p - 1) * (a + 1.0);
      if (k > 0)
        result.col(base_col + p)
            += 2 * k * result.col(base_col - m + p - 1) * (a + 1.0);
      if (p > 1)
        result.col(base_col + p) -= result.col(base_col + p - 2) * a;
    }
  }

  // Normalise
  for (int k = 0; k < nderiv + 1; ++k)
  {
    for (int p = 0; p < degree + 1; ++p)
      result.col(p + k * m) *= std::sqrt(p + 0.5);
  }

  return result;
}
//-----------------------------------------------------------------------------
// Compute the complete set of derivatives from 0 to nderiv, for all the
// polynomials up to order n on a triangle in [0, 1][0, 1].
// The polynomials P_{pq} are built up in sequence, firstly along q = 0, which
// is a line segment, as in tabulate_polyset_interval_derivs above, but with a
// change of variables. The polynomials are then extended in the q direction,
// using the relation given in Sherwin and Karniadakis 1995
// (https://doi.org/10.1016/0045-7825(94)00745-9)
Eigen::ArrayXXd tabulate_polyset_triangle_derivs(int n, int nderiv,
                                                 const Eigen::ArrayXXd& pts)

{
  assert(pts.cols() == 2);

  Eigen::ArrayXXd x = pts * 2.0 - 1.0;

  const int m = (n + 1) * (n + 2) / 2;
  const int md = (nderiv + 1) * (nderiv + 2) / 2;
  Eigen::ArrayXXd result(pts.rows(), m * md);

  // f3 = ((1-y)/2)^2
  const Eigen::ArrayXd f3 = (1.0 - x.col(1)).square() * 0.25;

  // Iterate over derivatives in increasing order, since higher derivatives
  // depend on earlier calculations
  for (int k = 0; k < nderiv + 1; ++k)
  {
    for (int kx = 0; kx < k + 1; ++kx)
    {
      const int ky = k - kx;
      const int base_col = idx(kx, ky) * m;

      if (kx == 0 and ky == 0)
        result.col(base_col).fill(1.0);
      else
        result.col(base_col).setZero();

      for (int p = 1; p < n + 1; ++p)
      {
        const double a
            = static_cast<double>(2 * p - 1) / static_cast<double>(p);
        result.col(base_col + idx(p, 0))
            = (x.col(0) + 0.5 * x.col(1) + 0.5)
              * result.col(base_col + idx(p - 1, 0)) * a;
        if (kx > 0)
        {
          result.col(base_col + idx(p, 0))
              += 2 * kx * a * result.col(idx(kx - 1, ky) * m + idx(p - 1, 0));
        }

        if (ky > 0)
        {
          result.col(base_col + idx(p, 0))
              += ky * a * result.col(idx(kx, ky - 1) * m + idx(p - 1, 0));
        }

        if (p > 1)
        {
          // y^2 terms
          result.col(base_col + idx(p, 0))
              -= f3 * result.col(base_col + idx(p - 2, 0)) * (a - 1.0);

          if (ky > 0)
          {
            result.col(base_col + idx(p, 0))
                -= ky * (x.col(1) - 1.0)
                   * result.col(idx(kx, ky - 1) * m + idx(p - 2, 0))
                   * (a - 1.0);
          }

          if (ky > 1)
          {
            result.col(base_col + idx(p, 0))
                -= ky * (ky - 1)
                   * result.col(idx(kx, ky - 2) * m + idx(p - 2, 0))
                   * (a - 1.0);
          }
        }
      }

      for (int p = 0; p < n; ++p)
      {
        result.col(base_col + idx(p, 1)) = result.col(base_col + idx(p, 0))
                                           * (x.col(1) * (1.5 + p) + 0.5 + p);
        if (ky > 0)
        {
          result.col(base_col + idx(p, 1))
              += 2 * ky * (1.5 + p)
                 * result.col(idx(kx, ky - 1) * m + idx(p, 0));
        }

        for (int q = 1; q < n - p; ++q)
        {
          const auto [a1, a2, a3] = jrc(2 * p + 1, q);
          result.col(base_col + idx(p, q + 1))
              = result.col(base_col + idx(p, q)) * (x.col(1) * a1 + a2)
                - result.col(base_col + idx(p, q - 1)) * a3;
          if (ky > 0)
          {
            result.col(base_col + idx(p, q + 1))
                += 2 * ky * a1 * result.col(idx(kx, ky - 1) * m + idx(p, q));
          }
        }
      }
    }
  }

  // Normalisation
  for (int j = 0; j < md; ++j)
  {
    for (int p = 0; p < n + 1; ++p)
      for (int q = 0; q < n - p + 1; ++q)
        result.col(m * j + idx(p, q)) *= std::sqrt((p + 0.5) * (p + q + 1));
  }

  return result;
}
//-----------------------------------------------------------------------------
Eigen::ArrayXXd tabulate_polyset_tetrahedron_derivs(int n, int nderiv,
                                                    const Eigen::ArrayXXd& pts)
{
  assert(pts.cols() == 3);

  Eigen::ArrayXXd x = pts * 2.0 - 1.0;

  const int m = (n + 1) * (n + 2) * (n + 3) / 6;
  const int md = (nderiv + 1) * (nderiv + 2) * (nderiv + 3) / 6;
  Eigen::ArrayXXd result(x.rows(), md * m);

  const Eigen::ArrayXd f2 = (x.col(1) + x.col(2)).square() * 0.25;
  const Eigen::ArrayXd f3 = (1.0 + x.col(1) * 2.0 + x.col(2)) * 0.5;
  const Eigen::ArrayXd f4 = (1.0 - x.col(2)) * 0.5;
  const Eigen::ArrayXd f5 = f4 * f4;

  // Traverse derivatives in increasing order
  for (int k = 0; k < nderiv + 1; ++k)
  {
    for (int j = 0; j < k + 1; ++j)
    {
      for (int kx = 0; kx < j + 1; ++kx)
      {
        const int ky = j - kx;
        const int kz = k - j;
        const int base_col = idx(kx, ky, kz) * m;
        if (kx == 0 and ky == 0 and kz == 0)
          result.col(base_col).fill(1.0);
        else
          result.col(base_col).setZero();

        for (int p = 1; p < n + 1; ++p)
        {
          double a = static_cast<double>(2 * p - 1) / static_cast<double>(p);
          result.col(base_col + idx(p, 0, 0))
              = (x.col(0) + 0.5 * (x.col(1) + x.col(2)) + 1.0)
                * result.col(base_col + idx(p - 1, 0, 0)) * a;
          if (kx > 0)
          {
            result.col(base_col + idx(p, 0, 0))
                += 2 * kx * a
                   * result.col(idx(kx - 1, ky, kz) * m + idx(p - 1, 0, 0));
          }

          if (ky > 0)
          {
            result.col(base_col + idx(p, 0, 0))
                += ky * a
                   * result.col(idx(kx, ky - 1, kz) * m + idx(p - 1, 0, 0));
          }

          if (kz > 0)
          {
            result.col(base_col + idx(p, 0, 0))
                += kz * a
                   * result.col(idx(kx, ky, kz - 1) * m + idx(p - 1, 0, 0));
          }

          if (p > 1)
          {
            result.col(base_col + idx(p, 0, 0))
                -= f2 * result.col(base_col + idx(p - 2, 0, 0)) * (a - 1.0);
            if (ky > 0)
            {
              result.col(base_col + idx(p, 0, 0))
                  -= ky * (x.col(1) + x.col(2))
                     * result.col(idx(kx, ky - 1, kz) * m + idx(p - 2, 0, 0))
                     * (a - 1.0);
            }

            if (ky > 1)
            {
              result.col(base_col + idx(p, 0, 0))
                  -= ky * (ky - 1)
                     * result.col(idx(kx, ky - 2, kz) * m + idx(p - 2, 0, 0))
                     * (a - 1.0);
            }

            if (kz > 0)
            {
              result.col(base_col + idx(p, 0, 0))
                  -= kz * (x.col(1) + x.col(2))
                     * result.col(idx(kx, ky, kz - 1) * m + idx(p - 2, 0, 0))
                     * (a - 1.0);
            }

            if (kz > 1)
            {
              result.col(base_col + idx(p, 0, 0))
                  -= kz * (kz - 1)
                     * result.col(idx(kx, ky, kz - 2) * m + idx(p - 2, 0, 0))
                     * (a - 1.0);
            }

            if (ky > 0 and kz > 0)
            {
              result.col(base_col + idx(p, 0, 0))
                  -= 2.0 * ky * kz
                     * result.col(idx(kx, ky - 1, kz - 1) * m
                                  + idx(p - 2, 0, 0))
                     * (a - 1.0);
            }
          }
        }

        for (int p = 0; p < n; ++p)
        {
          result.col(base_col + idx(p, 1, 0))
              = result.col(base_col + idx(p, 0, 0))
                * ((1.0 + x.col(1)) * p
                   + (2.0 + x.col(1) * 3.0 + x.col(2)) * 0.5);
          if (ky > 0)
          {
            result.col(base_col + idx(p, 1, 0))
                += 2 * ky * result.col(idx(kx, ky - 1, kz) * m + idx(p, 0, 0))
                   * (1.5 + p);
          }

          if (kz > 0)
          {
            result.col(base_col + idx(p, 1, 0))
                += kz * result.col(idx(kx, ky, kz - 1) * m + idx(p, 0, 0));
          }

          for (int q = 1; q < n - p; ++q)
          {
            auto [aq, bq, cq] = jrc(2 * p + 1, q);
            result.col(base_col + idx(p, q + 1, 0))
                = result.col(base_col + idx(p, q, 0)) * (f3 * aq + f4 * bq)
                  - result.col(base_col + idx(p, q - 1, 0)) * f5 * cq;

            if (ky > 0)
            {
              result.col(base_col + idx(p, q + 1, 0))
                  += 2 * ky * result.col(idx(kx, ky - 1, kz) * m + idx(p, q, 0))
                     * aq;
            }

            if (kz > 0)
            {
              result.col(base_col + idx(p, q + 1, 0))
                  += kz * result.col(idx(kx, ky, kz - 1) * m + idx(p, q, 0))
                         * (aq - bq)
                     + kz * (1.0 - x.col(2))
                           * result.col(idx(kx, ky, kz - 1) * m
                                        + idx(p, q - 1, 0))
                           * cq;
            }

            if (kz > 1)
            {
              // Quadratic term in z
              result.col(base_col + idx(p, q + 1, 0))
                  -= kz * (kz - 1)
                     * result.col(idx(kx, ky, kz - 2) * m + idx(p, q - 1, 0))
                     * cq;
            }
          }
        }

        for (int p = 0; p < n; ++p)
        {
          for (int q = 0; q < n - p; ++q)
          {
            result.col(base_col + idx(p, q, 1))
                = result.col(base_col + idx(p, q, 0))
                  * ((1.0 + p + q) + x.col(2) * (2.0 + p + q));
            if (kz > 0)
            {
              result.col(base_col + idx(p, q, 1))
                  += 2 * kz * (2.0 + p + q)
                     * result.col(idx(kx, ky, kz - 1) * m + idx(p, q, 0));
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
              result.col(base_col + idx(p, q, r + 1))
                  = result.col(base_col + idx(p, q, r)) * (x.col(2) * ar + br)
                    - result.col(base_col + idx(p, q, r - 1)) * cr;
              if (kz > 0)
              {
                result.col(base_col + idx(p, q, r + 1))
                    += 2 * kz * ar
                       * result.col(idx(kx, ky, kz - 1) * m + idx(p, q, r));
              }
            }
          }
        }
      }
    }
  }

  for (int i = 0; i < md; ++i)
  {
    for (int p = 0; p < n + 1; ++p)
    {
      for (int q = 0; q < n - p + 1; ++q)
      {
        for (int r = 0; r < n - p - q + 1; ++r)
        {
          result.col(i * m + idx(p, q, r))
              *= std::sqrt((p + 0.5) * (p + q + 1.0) * (p + q + r + 1.5));
        }
      }
    }
  }

  return result;
}
//-----------------------------------------------------------------------------
Eigen::ArrayXXd tabulate_polyset_pyramid_derivs(int n, int nderiv,
                                                const Eigen::ArrayXXd& pts)
{
  assert(pts.cols() == 3);

  Eigen::ArrayXXd x = pts * 2.0 - 1.0;

  const int m = (n + 1) * (n + 2) * (2 * n + 3) / 6;
  const int md = (nderiv + 1) * (nderiv + 2) * (nderiv + 3) / 6;
  Eigen::ArrayXXd result(x.rows(), md * m);

  // Indexing for pyramidal basis functions
  auto pyr_idx = [&n](int p, int q, int r) -> int {
    const int rv = n - r + 1;
    const int r0 = r * (n + 1) * (n - r + 2) + (2 * r - 1) * (r - 1) * r / 6;
    return r0 + p * rv + q;
  };

  const Eigen::ArrayXd f2 = (1.0 - x.col(2)).square() * 0.25;

  // Traverse derivatives in increasing order
  for (int k = 0; k < nderiv + 1; ++k)
  {
    for (int j = 0; j < k + 1; ++j)
    {
      for (int kx = 0; kx < j + 1; ++kx)
      {
        const int ky = j - kx;
        const int kz = k - j;
        const int base_col = idx(kx, ky, kz) * m;

        const int pyramidal_index = pyr_idx(0, 0, 0);
        assert(pyramidal_index < m);
        if (kx == 0 and ky == 0 and kz == 0)
          result.col(base_col + pyramidal_index).fill(1.0);
        else
          result.col(base_col + pyramidal_index).setZero();

        // r = 0
        for (int p = 0; p < n + 1; ++p)
        {
          if (p > 0)
          {
            const double a
                = static_cast<double>(p - 1) / static_cast<double>(p);
            result.col(base_col + pyr_idx(p, 0, 0))
                = (0.5 + x.col(0) + x.col(2) * 0.5)
                  * result.col(base_col + pyr_idx(p - 1, 0, 0)) * (a + 1.0);
            if (kx > 0)
            {
              result.col(base_col + pyr_idx(p, 0, 0))
                  += 2.0 * kx
                     * result.col(idx(kx - 1, ky, kz) * m
                                  + pyr_idx(p - 1, 0, 0))
                     * (a + 1.0);
            }

            if (kz > 0)
            {
              result.col(base_col + pyr_idx(p, 0, 0))
                  += kz
                     * result.col(idx(kx, ky, kz - 1) * m
                                  + pyr_idx(p - 1, 0, 0))
                     * (a + 1.0);
            }

            if (p > 1)
            {
              result.col(base_col + pyr_idx(p, 0, 0))
                  -= f2 * result.col(base_col + pyr_idx(p - 2, 0, 0)) * a;

              if (kz > 0)
              {
                result.col(base_col + pyr_idx(p, 0, 0))
                    += kz * (1.0 - x.col(2))
                       * result.col(idx(kx, ky, kz - 1) * m
                                    + pyr_idx(p - 2, 0, 0))
                       * a;
              }

              if (kz > 1)
              {
                // quadratic term in z
                result.col(base_col + pyr_idx(p, 0, 0))
                    -= kz * (kz - 1)
                       * result.col(idx(kx, ky, kz - 2) * m
                                    + pyr_idx(p - 2, 0, 0))
                       * a;
              }
            }
          }

          for (int q = 1; q < n + 1; ++q)
          {
            const double a
                = static_cast<double>(q - 1) / static_cast<double>(q);
            result.col(base_col + pyr_idx(p, q, 0))
                = (0.5 + x.col(1) + x.col(2) * 0.5)
                  * result.col(base_col + pyr_idx(p, q - 1, 0)) * (a + 1.0);
            if (ky > 0)
            {
              result.col(base_col + pyr_idx(p, q, 0))
                  += 2.0 * ky
                     * result.col(idx(kx, ky - 1, kz) * m
                                  + pyr_idx(p, q - 1, 0))
                     * (a + 1.0);
            }

            if (kz > 0)
            {
              result.col(base_col + pyr_idx(p, q, 0))
                  += kz
                     * result.col(idx(kx, ky, kz - 1) * m
                                  + pyr_idx(p, q - 1, 0))
                     * (a + 1.0);
            }

            if (q > 1)
            {
              result.col(base_col + pyr_idx(p, q, 0))
                  -= f2 * result.col(base_col + pyr_idx(p, q - 2, 0)) * a;

              if (kz > 0)
              {
                result.col(base_col + pyr_idx(p, q, 0))
                    += kz * (1.0 - x.col(2))
                       * result.col(idx(kx, ky, kz - 1) * m
                                    + pyr_idx(p, q - 2, 0))
                       * a;
              }

              if (kz > 1)
              {
                result.col(base_col + pyr_idx(p, q, 0))
                    -= kz * (kz - 1)
                       * result.col(idx(kx, ky, kz - 2) * m
                                    + pyr_idx(p, q - 2, 0))
                       * a;
              }
            }
          }
        }

        // Extend into r > 0
        for (int p = 0; p < n; ++p)
        {
          for (int q = 0; q < n; ++q)
          {
            result.col(base_col + pyr_idx(p, q, 1))
                = result.col(base_col + pyr_idx(p, q, 0))
                  * ((1.0 + p + q) + x.col(2) * (2.0 + p + q));
            if (kz > 0)
            {
              result.col(base_col + pyr_idx(p, q, 1))
                  += 2 * kz
                     * result.col(idx(kx, ky, kz - 1) * m + pyr_idx(p, q, 0))
                     * (2.0 + p + q);
            }
          }
        }

        for (int r = 1; r < n + 1; ++r)
        {
          for (int p = 0; p < n - r; ++p)
          {
            for (int q = 0; q < n - r; ++q)
            {
              auto [ar, br, cr] = jrc(2 * p + 2 * q + 2, r);
              result.col(base_col + pyr_idx(p, q, r + 1))
                  = result.col(base_col + pyr_idx(p, q, r))
                        * (x.col(2) * ar + br)
                    - result.col(base_col + pyr_idx(p, q, r - 1)) * cr;
              if (kz > 0)
              {
                result.col(base_col + pyr_idx(p, q, r + 1))
                    += ar * 2 * kz
                       * result.col(idx(kx, ky, kz - 1) * m + pyr_idx(p, q, r));
              }
            }
          }
        }
      }
    }
  }

  for (int i = 0; i < md; ++i)
  {
    for (int r = 0; r < n + 1; ++r)
    {
      for (int p = 0; p < n - r + 1; ++p)
      {
        for (int q = 0; q < n - r + 1; ++q)
        {
          result.col(i * m + pyr_idx(p, q, r))
              *= std::sqrt((q + 0.5) * (p + 0.5) * (p + q + r + 1.5));
        }
      }
    }
  }

  return result;
}
//-----------------------------------------------------------------------------
Eigen::ArrayXXd tabulate_polyset_quad_derivs(int n, int nderiv,
                                             const Eigen::ArrayXXd& pts)
{
  assert(pts.cols() == 2);
  const int m = (n + 1) * (n + 1);
  const int md = (nderiv + 1) * (nderiv + 2) / 2;

  Eigen::ArrayXXd px = tabulate_polyset_line_derivs(n, nderiv, pts.col(0));
  Eigen::ArrayXXd py = tabulate_polyset_line_derivs(n, nderiv, pts.col(1));

  Eigen::ArrayXXd result(pts.rows(), m * md);
  for (int kx = 0; kx < nderiv + 1; ++kx)
  {
    for (int ky = 0; ky < nderiv + 1 - kx; ++ky)
    {
      const int base_col = idx(kx, ky) * m;
      int c = 0;
      for (int i = 0; i < (n + 1); ++i)
        for (int j = 0; j < (n + 1); ++j)
          result.col(base_col + c++)
              = px.col(kx * (n + 1) + i) * py.col(ky * (n + 1) + j);
    }
  }

  return result;
}
//-----------------------------------------------------------------------------
Eigen::ArrayXXd tabulate_polyset_hex_derivs(int n, int nderiv,
                                            const Eigen::ArrayXXd& pts)
{
  assert(pts.cols() == 3);
  const int m = (n + 1) * (n + 1) * (n + 1);
  const int md = (nderiv + 1) * (nderiv + 2) * (nderiv + 3) / 6;

  Eigen::ArrayXXd px = tabulate_polyset_line_derivs(n, nderiv, pts.col(0));
  Eigen::ArrayXXd py = tabulate_polyset_line_derivs(n, nderiv, pts.col(1));
  Eigen::ArrayXXd pz = tabulate_polyset_line_derivs(n, nderiv, pts.col(2));

  Eigen::ArrayXXd result(pts.rows(), m * md);
  for (int kx = 0; kx < nderiv + 1; ++kx)
  {
    for (int ky = 0; ky < nderiv + 1 - kx; ++ky)
    {
      for (int kz = 0; kz < nderiv + 1 - kx - ky; ++kz)
      {
        const int base_col = idx(kx, ky, kz) * m;
        int c = 0;
        for (int i = 0; i < (n + 1); ++i)
          for (int j = 0; j < (n + 1); ++j)
            for (int k = 0; k < (n + 1); ++k)
              result.col(base_col + c++) = px.col(kx * (n + 1) + i)
                                           * py.col(ky * (n + 1) + j)
                                           * pz.col(kz * (n + 1) + k);
      }
    }
  }

  return result;
}
//-----------------------------------------------------------------------------
Eigen::ArrayXXd tabulate_polyset_prism_derivs(int n, int nderiv,
                                              const Eigen::ArrayXXd& pts)
{
  assert(pts.cols() == 3);
  const int m = (n + 1) * (n + 1) * (n + 2) / 2;
  const int md = (nderiv + 1) * (nderiv + 2) * (nderiv + 3) / 6;

  Eigen::ArrayXXd pxy
      = tabulate_polyset_triangle_derivs(n, nderiv, pts.leftCols(2));
  Eigen::ArrayXXd pz = tabulate_polyset_line_derivs(n, nderiv, pts.col(2));

  Eigen::ArrayXXd result(pts.rows(), m * md);
  for (int kx = 0; kx < nderiv + 1; ++kx)
  {
    for (int ky = 0; ky < nderiv + 1 - kx; ++ky)
    {
      for (int kz = 0; kz < nderiv + 1 - kx - ky; ++kz)
      {
        const int base_col = idx(kx, ky, kz) * m;
        int c = 0;
        for (int i = 0; i < (n + 1) * (n + 2) / 2; ++i)
          for (int k = 0; k < (n + 1); ++k)
            result.col(base_col + c++)
                = pxy.col(idx(kx, ky) * (n + 1) * (n + 2) / 2 + i)
                  * pz.col(kz * (n + 1) + k);
      }
    }
  }

  return result;
}
} // namespace
//-----------------------------------------------------------------------------
Eigen::ArrayXXd polyset::tabulate(cell::type celltype, int n, int nderiv,
                                  const Eigen::ArrayXXd& pts)
{
  switch (celltype)
  {
  case cell::type::interval:
    return tabulate_polyset_line_derivs(n, nderiv, pts);
  case cell::type::triangle:
    return tabulate_polyset_triangle_derivs(n, nderiv, pts);
  case cell::type::tetrahedron:
    return tabulate_polyset_tetrahedron_derivs(n, nderiv, pts);
  case cell::type::quadrilateral:
    return tabulate_polyset_quad_derivs(n, nderiv, pts);
  case cell::type::prism:
    return tabulate_polyset_prism_derivs(n, nderiv, pts);
  case cell::type::pyramid:
    return tabulate_polyset_pyramid_derivs(n, nderiv, pts);
  case cell::type::hexahedron:
    return tabulate_polyset_hex_derivs(n, nderiv, pts);
  default:
    throw std::runtime_error("Polynomial set: Unsupported cell type");
  }
}
//-----------------------------------------------------------------------------
int polyset::dim(cell::type celltype, int n)
{
  switch (celltype)
  {
  case cell::type::triangle:
    return (n + 1) * (n + 2) / 2;
  case cell::type::tetrahedron:
    return (n + 1) * (n + 2) * (n + 3) / 6;
  case cell::type::prism:
    return (n + 1) * (n + 1) * (n + 2) / 2;
  case cell::type::pyramid:
    return (n + 1) * (n + 2) * (2 * n + 3) / 6;
  case cell::type::interval:
    return (n + 1);
  case cell::type::quadrilateral:
    return (n + 1) * (n + 1);
  case cell::type::hexahedron:
    return (n + 1) * (n + 1) * (n + 1);
  default:
    return 1;
  }
}
//-----------------------------------------------------------------------------
