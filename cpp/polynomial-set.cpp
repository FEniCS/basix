// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "polynomial-set.h"
#include "cell.h"
#include <Eigen/Dense>

namespace
{
// Compute coefficients in the Jacobi Polynomial recurrence relation
std::tuple<double, double, double> jrc(int a, int n)
{
  double an = (a + 2 * n + 1) * (a + 2 * n + 2)
              / static_cast<double>(2 * (n + 1) * (a + n + 1));
  double bn = a * a * (a + 2 * n + 1)
              / static_cast<double>(2 * (n + 1) * (a + n + 1) * (a + 2 * n));
  double cn = n * (a + n) * (a + 2 * n + 2)
              / static_cast<double>((n + 1) * (a + n + 1) * (a + 2 * n));
  return std::tuple<double, double, double>(an, bn, cn);
}
//-----------------------------------------------------------------------------
// Compute the complete set of derivatives from 0 to nderiv, for all the
// polynomials up to order n on a line segment. The polynomials used are
// Legendre Polynomials, with the recurrence relation given by
// n P(n) = (2n - 1) x P_{n-1} - (n - 1) P_{n-2} in the interval [-1, 1]. The
// range is rescaled here to [0, 1].
std::vector<
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
tabulate_polyset_line_derivs(
    int n, int nderiv,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        pts)
{
  assert(pts.cols() == 1);

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x
      = pts * 2.0 - 1.0;

  const int m = (n + 1);

  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      dresult(nderiv + 1);

  for (int k = 0; k < nderiv + 1; ++k)
  {
    // Reference to this derivative, and resize
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        result
        = dresult[k];
    result.resize(pts.rows(), m);

    if (k == 0)
      result.col(0).fill(1.0);
    else
      result.col(0).setZero();

    for (int p = 1; p < n + 1; ++p)
    {
      double a = 1.0 - 1.0 / static_cast<double>(p);
      result.col(p) = x * result.col(p - 1) * (a + 1.0);
      if (k > 0)
        result.col(p) += 2 * k * dresult[k - 1].col(p - 1) * (a + 1.0);
      if (p > 1)
        result.col(p) -= result.col(p - 2) * a;
    }
  }

  // Normalise
  for (int k = 0; k < nderiv + 1; ++k)
  {
    for (int p = 0; p < n + 1; ++p)
      dresult[k].col(p) *= sqrt(p + 0.5);
  }

  return dresult;
}
//-----------------------------------------------------------------------------
// Compute the complete set of derivatives from 0 to nderiv, for all the
// polynomials up to order n on a triangle in [0, 1][0, 1].
// The polynomials P_{pq} are built up in sequence, firstly along q = 0, which
// is a line segment, as in tabulate_polyset_interval_derivs above, but with a
// change of variables. The polynomials are then extended in the q direction,
// using the relation given in Sherwin and Karniadakis 1995
// (https://doi.org/10.1016/0045-7825(94)00745-9)
std::vector<
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
tabulate_polyset_triangle_derivs(
    int n, int nderiv,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        pts)

{
  assert(pts.cols() == 2);

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x
      = pts * 2.0 - 1.0;

  const int m = (n + 1) * (n + 2) / 2;
  const int md = (nderiv + 1) * (nderiv + 2) / 2;
  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      dresult(md);

  // f3 = ((1-y)/2)^2
  const auto f3 = (1.0 - x.col(1)).square() * 0.25;

  // Iterate over derivatives in increasing order, since higher derivatives
  // depend on earlier calculations
  for (int k = 0; k < nderiv + 1; ++k)
  {
    for (int kx = 0; kx < k + 1; ++kx)
    {
      const int ky = k - kx;

      // Get reference to this derivative and resize
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
          result
          = dresult[idx(kx, ky)];
      result.resize(pts.rows(), m);

      if (kx == 0 and ky == 0)
        result.col(0).fill(1.0);
      else
        result.col(0).setZero();

      for (int p = 1; p < n + 1; ++p)
      {
        const double a
            = static_cast<double>(2 * p - 1) / static_cast<double>(p);
        result.col(idx(p, 0))
            = (x.col(0) + 0.5 * x.col(1) + 0.5) * result.col(idx(p - 1, 0)) * a;
        if (kx > 0)
          result.col(idx(p, 0))
              += 2 * kx * a * dresult[idx(kx - 1, ky)].col(idx(p - 1, 0));
        if (ky > 0)
          result.col(idx(p, 0))
              += ky * a * dresult[idx(kx, ky - 1)].col(idx(p - 1, 0));
        if (p > 1)
        {
          // y^2 terms
          result.col(idx(p, 0)) -= f3 * result.col(idx(p - 2, 0)) * (a - 1.0);

          if (ky > 0)
          {
            result.col(idx(p, 0))
                -= ky * (x.col(1) - 1.0)
                   * dresult[idx(kx, ky - 1)].col(idx(p - 2, 0)) * (a - 1.0);
          }

          if (ky > 1)
          {
            result.col(idx(p, 0))
                -= ky * (ky - 1) * dresult[idx(kx, ky - 2)].col(idx(p - 2, 0))
                   * (a - 1.0);
          }
        }
      }

      for (int p = 0; p < n; ++p)
      {
        result.col(idx(p, 1))
            = result.col(idx(p, 0)) * (x.col(1) * (1.5 + p) + 0.5 + p);
        if (ky > 0)
          result.col(idx(p, 1))
              += 2 * ky * (1.5 + p) * dresult[idx(kx, ky - 1)].col(idx(p, 0));
        for (int q = 1; q < n - p; ++q)
        {
          auto [a1, a2, a3] = jrc(2 * p + 1, q);
          result.col(idx(p, q + 1))
              = result.col(idx(p, q)) * (x.col(1) * a1 + a2)
                - result.col(idx(p, q - 1)) * a3;
          if (ky > 0)
          {
            result.col(idx(p, q + 1))
                += 2 * ky * a1 * dresult[idx(kx, ky - 1)].col(idx(p, q));
          }
        }
      }
    }
  }

  // Normalisation
  for (std::size_t j = 0; j < dresult.size(); ++j)
  {
    for (int p = 0; p < n + 1; ++p)
      for (int q = 0; q < n - p + 1; ++q)
        dresult[j].col(idx(p, q)) *= sqrt((p + 0.5) * (p + q + 1));
  }

  return dresult;
}
//-----------------------------------------------------------------------------
std::vector<
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
tabulate_polyset_tetrahedron_derivs(
    int n, int nderiv,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        pts)
{
  assert(pts.cols() == 3);

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x
      = pts * 2.0 - 1.0;

  const int m = (n + 1) * (n + 2) * (n + 3) / 6;
  const int md = (nderiv + 1) * (nderiv + 2) * (nderiv + 3) / 6;
  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      dresult(md);

  const auto f2 = (x.col(1) + x.col(2)).square() * 0.25;
  const auto f3 = (1.0 + x.col(1) * 2.0 + x.col(2)) * 0.5;
  const auto f4 = (1.0 - x.col(2)) * 0.5;
  const auto f5 = f4 * f4;

  // Traverse derivatives in increasing order
  for (int k = 0; k < nderiv + 1; ++k)
  {
    for (int j = 0; j < k + 1; ++j)
    {
      for (int kx = 0; kx < j + 1; ++kx)
      {
        const int ky = j - kx;
        const int kz = k - j;

        Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
            result
            = dresult[idx(kx, ky, kz)];
        result.resize(pts.rows(), m);

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
            result.col(idx(p, 0, 0))
                += 2 * kx * a
                   * dresult[idx(kx - 1, ky, kz)].col(idx(p - 1, 0, 0));
          if (ky > 0)
            result.col(idx(p, 0, 0))
                += ky * a * dresult[idx(kx, ky - 1, kz)].col(idx(p - 1, 0, 0));
          if (kz > 0)
            result.col(idx(p, 0, 0))
                += kz * a * dresult[idx(kx, ky, kz - 1)].col(idx(p - 1, 0, 0));

          if (p > 1)
          {
            result.col(idx(p, 0, 0))
                -= f2 * result.col(idx(p - 2, 0, 0)) * (a - 1.0);
            // FIXME: y^2 and z^2 derivs
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

        for (int p = 0; p < n - 1; ++p)
          for (int q = 0; q < n - p - 1; ++q)
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
  }

  for (auto& result : dresult)
    for (int p = 0; p < n + 1; ++p)
      for (int q = 0; q < n - p + 1; ++q)
        for (int r = 0; r < n - p - q + 1; ++r)
          result.col(idx(p, q, r))
              *= sqrt((p + 0.5) * (p + q + 1.0) * (p + q + r + 1.5));

  return dresult;
}
//-----------------------------------------------------------------------------
std::vector<
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
tabulate_polyset_pyramid_derivs(
    int n, int nderiv,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        pts)
{
  assert(pts.cols() == 3);

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x
      = pts * 2.0 - 1.0;

  const int m = (n + 1) * (n + 2) * (2 * n + 3) / 6;
  const int md = (nderiv + 1) * (nderiv + 2) * (nderiv + 3) / 6;
  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      dresult(md);

  // Indexing for pyramidal basis functions
  auto pyr_idx = [&n, &m](int p, int q, int r) {
    const int rv = n - r + 1;
    const int r0 = r * (n + 1) * (n - r + 2) + (2 * r - 1) * (r - 1) * r / 6;
    const int idx = r0 + p * rv + q;
    assert(idx < m);
    return idx;
  };

  const auto f2 = (1.0 - x.col(2)).square() * 0.25;

  // Traverse derivatives in increasing order
  for (int k = 0; k < nderiv + 1; ++k)
  {
    for (int j = 0; j < k + 1; ++j)
    {
      for (int kx = 0; kx < j + 1; ++kx)
      {
        const int ky = j - kx;
        const int kz = k - j;
        Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
            result
            = dresult[idx(kx, ky, kz)];
        result.resize(pts.rows(), m);
        result.setZero();

        if (kx == 0 and ky == 0 and kz == 0)
          result.col(pyr_idx(0, 0, 0)).fill(1.0);
        else
          result.col(pyr_idx(0, 0, 0)).setZero();

        // r = 0
        for (int p = 0; p < n + 1; ++p)
        {
          if (p > 0)
          {
            const double a
                = static_cast<double>(p - 1) / static_cast<double>(p);
            result.col(pyr_idx(p, 0, 0)) = (0.5 + x.col(0) + x.col(2) * 0.5)
                                           * result.col(pyr_idx(p - 1, 0, 0))
                                           * (a + 1.0);
            if (kx > 0)
              result.col(pyr_idx(p, 0, 0))
                  += 2.0 * kx
                     * dresult[idx(kx - 1, ky, kz)].col(pyr_idx(p - 1, 0, 0))
                     * (a + 1.0);
            if (kz > 0)
              result.col(pyr_idx(p, 0, 0))
                  += kz * dresult[idx(kx, ky, kz - 1)].col(pyr_idx(p - 1, 0, 0))
                     * (a + 1.0);

            if (p > 1)
            {
              result.col(pyr_idx(p, 0, 0))
                  -= f2 * result.col(pyr_idx(p - 2, 0, 0)) * a;

              if (kz > 0)
                result.col(pyr_idx(p, 0, 0))
                    += kz * (1.0 - x.col(2))
                       * dresult[idx(kx, ky, kz - 1)].col(pyr_idx(p - 2, 0, 0))
                       * a;
              if (kz > 1)
              {
                // quadratic term in z
                result.col(pyr_idx(p, 0, 0))
                    -= kz * (kz - 1)
                       * dresult[idx(kx, ky, kz - 2)].col(pyr_idx(p - 2, 0, 0))
                       * a;
              }
            }
          }

          for (int q = 1; q < n + 1; ++q)
          {
            const double a
                = static_cast<double>(q - 1) / static_cast<double>(q);
            result.col(pyr_idx(p, q, 0)) = (0.5 + x.col(1) + x.col(2) * 0.5)
                                           * result.col(pyr_idx(p, q - 1, 0))
                                           * (a + 1.0);
            if (ky > 0)
              result.col(pyr_idx(p, q, 0))
                  += 2.0 * ky
                     * dresult[idx(kx, ky - 1, kz)].col(pyr_idx(p, q - 1, 0))
                     * (a + 1.0);
            if (kz > 0)
              result.col(pyr_idx(p, q, 0))
                  += kz * dresult[idx(kx, ky, kz - 1)].col(pyr_idx(p, q - 1, 0))
                     * (a + 1.0);
            if (q > 1)
            {
              result.col(pyr_idx(p, q, 0))
                  -= f2 * result.col(pyr_idx(p, q - 2, 0)) * a;

              if (kz > 0)
                result.col(pyr_idx(p, q, 0))
                    += kz * (1.0 - x.col(2))
                       * dresult[idx(kx, ky, kz - 1)].col(pyr_idx(p, q - 2, 0))
                       * a;
              if (kz > 1)
              {
                result.col(pyr_idx(p, q, 0))
                    -= kz * (kz - 1)
                       * dresult[idx(kx, ky, kz - 2)].col(pyr_idx(p, q - 2, 0))
                       * a;
              }
            }
          }
        }

        // Extend into r > 0
        for (int p = 0; p < n; ++p)
          for (int q = 0; q < n; ++q)
          {
            result.col(pyr_idx(p, q, 1))
                = result.col(pyr_idx(p, q, 0))
                  * ((1.0 + p + q) + x.col(2) * (2.0 + p + q));
            if (kz > 0)
              result.col(pyr_idx(p, q, 1))
                  += 2 * kz * dresult[idx(kx, ky, kz - 1)].col(pyr_idx(p, q, 0))
                     * (2.0 + p + q);
          }

        for (int r = 1; r < n + 1; ++r)
          for (int p = 0; p < n - r; ++p)
            for (int q = 0; q < n - r; ++q)
            {
              auto [ar, br, cr] = jrc(2 * p + 2 * q + 2, r);
              result.col(pyr_idx(p, q, r + 1))
                  = result.col(pyr_idx(p, q, r)) * (x.col(2) * ar + br)
                    - result.col(pyr_idx(p, q, r - 1)) * cr;
              if (kz > 0)
                result.col(pyr_idx(p, q, r + 1))
                    += ar * 2 * kz
                       * dresult[idx(kx, ky, kz - 1)].col(pyr_idx(p, q, r));
            }
      }
    }
  }

  for (auto& result : dresult)
    for (int r = 0; r < n + 1; ++r)
      for (int p = 0; p < n - r + 1; ++p)
        for (int q = 0; q < n - r + 1; ++q)
        {
          result.col(pyr_idx(p, q, r))
              *= sqrt((q + 0.5) * (p + 0.5) * (p + q + r + 1.5));
        }

  return dresult;
}
//-----------------------------------------------------------------------------
std::vector<
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
tabulate_polyset_quad_derivs(
    int n, int nderiv,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        pts)
{
  assert(pts.cols() == 2);
  const int m = (n + 1) * (n + 1);
  const int md = (nderiv + 1) * (nderiv + 2) / 2;

  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      dresult(md);
  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      px = tabulate_polyset_line_derivs(n, nderiv, pts.col(0));
  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      py = tabulate_polyset_line_derivs(n, nderiv, pts.col(1));

  for (int kx = 0; kx < nderiv + 1; ++kx)
    for (int ky = 0; ky < nderiv + 1 - kx; ++ky)
    {
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
          result
          = dresult[idx(kx, ky)];
      result.resize(pts.rows(), m);

      int c = 0;
      for (int i = 0; i < px[kx].cols(); ++i)
        for (int j = 0; j < py[ky].cols(); ++j)
          result.col(c++) = px[kx].col(i) * py[ky].col(j);
    }

  return dresult;
}
//-----------------------------------------------------------------------------
std::vector<
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
tabulate_polyset_hex_derivs(
    int n, int nderiv,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        pts)
{
  assert(pts.cols() == 3);
  const int m = (n + 1) * (n + 1) * (n + 1);

  const int md = (nderiv + 1) * (nderiv + 2) * (nderiv + 3) / 6;

  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      dresult(md);

  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      px = tabulate_polyset_line_derivs(n, nderiv, pts.col(0));
  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      py = tabulate_polyset_line_derivs(n, nderiv, pts.col(1));
  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      pz = tabulate_polyset_line_derivs(n, nderiv, pts.col(2));

  for (int kx = 0; kx < nderiv + 1; ++kx)
    for (int ky = 0; ky < nderiv + 1 - kx; ++ky)
      for (int kz = 0; kz < nderiv + 1 - kx - ky; ++kz)
      {
        Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
            result
            = dresult[idx(kx, ky, kz)];
        result.resize(pts.rows(), m);

        int c = 0;
        for (int i = 0; i < px[kx].cols(); ++i)
          for (int j = 0; j < py[ky].cols(); ++j)
            for (int k = 0; k < pz[kz].cols(); ++k)
              result.col(c++) = px[kx].col(i) * py[ky].col(j) * pz[kz].col(k);
      }

  return dresult;
}
//-----------------------------------------------------------------------------
std::vector<
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
tabulate_polyset_prism_derivs(
    int n, int nderiv,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        pts)
{
  assert(pts.cols() == 3);
  const int m = (n + 1) * (n + 1) * (n + 2) / 2;
  const int md = (nderiv + 1) * (nderiv + 2) * (nderiv + 3) / 6;

  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      dresult(md);

  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      pxy = tabulate_polyset_triangle_derivs(n, nderiv, pts.leftCols(2));
  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      pz = tabulate_polyset_line_derivs(n, nderiv, pts.col(2));

  for (int kx = 0; kx < nderiv + 1; ++kx)
    for (int ky = 0; ky < nderiv + 1 - kx; ++ky)
      for (int kz = 0; kz < nderiv + 1 - kx - ky; ++kz)
      {
        Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
            result
            = dresult[idx(kx, ky, kz)];
        result.resize(pts.rows(), m);
        int c = 0;
        for (int i = 0; i < pxy[idx(kx, ky)].cols(); ++i)
          for (int k = 0; k < pz[kz].cols(); ++k)
            result.col(c++) = pxy[idx(kx, ky)].col(i) * pz[kz].col(k);
      }

  return dresult;
}
} // namespace
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
PolynomialSet::tabulate_polynomial_set(
    Cell::Type celltype, int n,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        pts)
{
  switch (celltype)
  {
  case Cell::Type::interval:
    return tabulate_polyset_line_derivs(n, 0, pts)[0];
  case Cell::Type::triangle:
    return tabulate_polyset_triangle_derivs(n, 0, pts)[0];
  case Cell::Type::tetrahedron:
    return tabulate_polyset_tetrahedron_derivs(n, 0, pts)[0];
  case Cell::Type::quadrilateral:
    return tabulate_polyset_quad_derivs(n, 0, pts)[0];
  case Cell::Type::hexahedron:
    return tabulate_polyset_hex_derivs(n, 0, pts)[0];
  case Cell::Type::prism:
    return tabulate_polyset_prism_derivs(n, 0, pts)[0];
  case Cell::Type::pyramid:
    return tabulate_polyset_pyramid_derivs(n, 0, pts)[0];
  default:
    throw std::runtime_error("Polynomial set: Unsupported cell type");
  }
}

//-----------------------------------------------------------------------------
std::vector<
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
PolynomialSet::tabulate_polynomial_set_deriv(
    Cell::Type celltype, int n, int nderiv,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        pts)
{
  switch (celltype)
  {
  case Cell::Type::interval:
    return tabulate_polyset_line_derivs(n, nderiv, pts);
  case Cell::Type::triangle:
    return tabulate_polyset_triangle_derivs(n, nderiv, pts);
  case Cell::Type::tetrahedron:
    return tabulate_polyset_tetrahedron_derivs(n, nderiv, pts);
  case Cell::Type::quadrilateral:
    return tabulate_polyset_quad_derivs(n, nderiv, pts);
  case Cell::Type::prism:
    return tabulate_polyset_prism_derivs(n, nderiv, pts);
  case Cell::Type::pyramid:
    return tabulate_polyset_pyramid_derivs(n, nderiv, pts);
  case Cell::Type::hexahedron:
    return tabulate_polyset_hex_derivs(n, nderiv, pts);
  default:
    throw std::runtime_error("Polynomial set: Unsupported cell type");
  }
}
