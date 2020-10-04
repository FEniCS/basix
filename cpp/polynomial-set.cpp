// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "polynomial-set.h"
#include "cell.h"
#include <Eigen/Dense>
#include <iostream>

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

  // Differentiate wrt x first
  for (int k = 0; k < nderiv + 1; ++k)
  {
    // Get reference to this derivative and resize
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        result
        = dresult[idx(k, 0)];
    result.resize(pts.rows(), m);

    if (k == 0)
      result.col(0).fill(1.0);
    else
      result.col(0).setZero();

    for (int p = 1; p < n + 1; ++p)
    {
      const double a = static_cast<double>(2 * p - 1) / static_cast<double>(p);
      result.col(idx(p, 0))
          = (x.col(0) + 0.5 * x.col(1) + 0.5) * result.col(idx(p - 1, 0)) * a;
      if (p > 1)
        result.col(idx(p, 0)) -= f3 * result.col(idx(p - 2, 0)) * (a - 1.0);
      if (k > 0)
        result.col(idx(p, 0))
            += 2 * k * a * dresult[idx(k - 1, 0)].col(idx(p - 1, 0));
    }

    for (int p = 0; p < n; ++p)
    {
      result.col(idx(p, 1))
          = result.col(idx(p, 0)) * (x.col(1) * (1.5 + p) + 0.5 + p);
      for (int q = 1; q < n - p; ++q)
      {
        auto [a1, a2, a3] = jrc(2 * p + 1, q);
        result.col(idx(p, q + 1)) = result.col(idx(p, q)) * (x.col(1) * a1 + a2)
                                    - result.col(idx(p, q - 1)) * a3;
      }
    }
  }

  // Now differentiate wrt y
  for (int ky = 1; ky < nderiv + 1; ++ky)
    for (int kx = 0; kx < (nderiv + 1 - ky); ++kx)
    {
      // Get reference to this derivative and resize
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
          result
          = dresult[idx(kx, ky)];
      result.resize(pts.rows(), m);

      result.col(0).setZero();

      for (int p = 1; p < n + 1; ++p)
      {
        const double a
            = static_cast<double>(2 * p - 1) / static_cast<double>(p);
        result.col(idx(p, 0))
            = (x.col(0) + 0.5 * x.col(1) + 0.5) * a * result.col(idx(p - 1, 0))
              + ky * a * dresult[idx(kx, ky - 1)].col(idx(p - 1, 0));

        if (p > 1)
        {
          // y^2 terms
          result.col(idx(p, 0))
              -= f3 * result.col(idx(p - 2, 0)) * (a - 1.0)
                 + ky * (x.col(1) - 1.0)
                       * dresult[idx(kx, ky - 1)].col(idx(p - 2, 0))
                       * (a - 1.0);
          if (ky > 1)
            result.col(idx(p, 0))
                -= ky * (ky - 1) * dresult[idx(kx, ky - 2)].col(idx(p - 2, 0))
                   * (a - 1.0);
        }
      }

      for (int p = 0; p < n; ++p)
      {
        result.col(idx(p, 1))
            = result.col(idx(p, 0)) * (x.col(1) * (1.5 + p) + 0.5 + p)
              + 2 * ky * (1.5 + p) * dresult[idx(kx, ky - 1)].col(idx(p, 0));
        for (int q = 1; q < n - p; ++q)
        {
          auto [a1, a2, a3] = jrc(2 * p + 1, q);
          result.col(idx(p, q + 1))
              = result.col(idx(p, q)) * (x.col(1) * a1 + a2)
                + 2 * ky * a1 * dresult[idx(kx, ky - 1)].col(idx(p, q))
                - result.col(idx(p, q - 1)) * a3;
        }
      }
    }

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

  // d/dx first
  for (int k = 0; k < nderiv + 1; ++k)
  {
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        result
        = dresult[idx(k, 0, 0)];
    result.resize(pts.rows(), m);

    if (k == 0)
      result.col(0).fill(1.0);
    else
      result.col(0).setZero();

    for (int p = 1; p < n + 1; ++p)
    {
      double a = static_cast<double>(2 * p - 1) / static_cast<double>(p);
      result.col(idx(p, 0, 0)) = (x.col(0) + 0.5 * (x.col(1) + x.col(2)) + 1.0)
                                 * result.col(idx(p - 1, 0, 0)) * a;
      if (p > 1)
        result.col(idx(p, 0, 0))
            -= f2 * result.col(idx(p - 2, 0, 0)) * (a - 1.0);
      if (k > 0)
        result.col(idx(p, 0, 0))
            += 2 * k * a * dresult[idx(k - 1, 0, 0)].col(idx(p - 1, 0, 0));
    }

    for (int p = 0; p < n; ++p)
    {
      result.col(idx(p, 1, 0))
          = result.col(idx(p, 0, 0))
            * ((1.0 + x.col(1)) * p + (2.0 + x.col(1) * 3.0 + x.col(2)) * 0.5);
      for (int q = 1; q < n - p; ++q)
      {
        auto [aq, bq, cq] = jrc(2 * p + 1, q);
        result.col(idx(p, q + 1, 0))
            = result.col(idx(p, q, 0)) * (f3 * aq + f4 * bq)
              - result.col(idx(p, q - 1, 0)) * f5 * cq;
      }
    }

    for (int p = 0; p < n; ++p)
      for (int q = 0; q < n - p; ++q)
        result.col(idx(p, q, 1)) = result.col(idx(p, q, 0))
                                   * ((1.0 + p + q) + x.col(2) * (2.0 + p + q));

    for (int p = 0; p < n - 1; ++p)
      for (int q = 0; q < n - p - 1; ++q)
        for (int r = 1; r < n - p - q; ++r)
        {
          auto [ar, br, cr] = jrc(2 * p + 2 * q + 2, r);
          result.col(idx(p, q, r + 1))
              = result.col(idx(p, q, r)) * (x.col(2) * ar + br)
                - result.col(idx(p, q, r - 1)) * cr;
        }
  }

  // Now differentiate wrt y
  for (int ky = 1; ky < nderiv + 1; ++ky)
    for (int kx = 0; kx < (nderiv + 1 - ky); ++kx)
    {
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
          result
          = dresult[idx(kx, ky, 0)];
      result.resize(pts.rows(), m);

      result.col(0).setZero();

      for (int p = 1; p < n + 1; ++p)
      {
        double a = static_cast<double>(2 * p - 1) / static_cast<double>(p);
        result.col(idx(p, 0, 0))
            = (x.col(0) + 0.5 * (x.col(1) + x.col(2)) + 1.0)
                  * result.col(idx(p - 1, 0, 0)) * a
              + ky * a * dresult[idx(kx, ky - 1, 0)].col(idx(p - 1, 0, 0));
        if (p > 1)
        {
          result.col(idx(p, 0, 0))
              -= f2 * result.col(idx(p - 2, 0, 0)) * (a - 1.0)
                 + ky * (x.col(1) + x.col(2))
                       * dresult[idx(kx, ky - 1, 0)].col(idx(p - 2, 0, 0))
                       * (a - 1.0);
          if (ky > 1)
            result.col(idx(p, 0, 0))
                -= ky * (ky - 1)
                   * dresult[idx(kx, ky - 2, 0)].col(idx(p - 2, 0, 0))
                   * (a - 1.0);
        }
      }

      for (int p = 0; p < n; ++p)
      {
        result.col(idx(p, 1, 0))
            = result.col(idx(p, 0, 0))
                  * (1.0 + p + x.col(1) * (1.5 + p) + x.col(2) * 0.5)
              + 2 * ky * dresult[idx(kx, ky - 1, 0)].col(idx(p, 0, 0))
                    * (1.5 + p);
        for (int q = 1; q < n - p; ++q)
        {
          auto [aq, bq, cq] = jrc(2 * p + 1, q);
          result.col(idx(p, q + 1, 0))
              = result.col(idx(p, q, 0)) * (f3 * aq + f4 * bq)
                - result.col(idx(p, q - 1, 0)) * f5 * cq
                + 2 * ky * dresult[idx(kx, ky - 1, 0)].col(idx(p, q, 0)) * aq;
        }
      }

      for (int p = 0; p < n; ++p)
        for (int q = 0; q < n - p; ++q)
          result.col(idx(p, q, 1))
              = result.col(idx(p, q, 0))
                * ((1.0 + p + q) + x.col(2) * (2.0 + p + q));

      for (int p = 0; p < n - 1; ++p)
        for (int q = 0; q < n - p - 1; ++q)
          for (int r = 1; r < n - p - q; ++r)
          {
            auto [ar, br, cr] = jrc(2 * p + 2 * q + 2, r);
            result.col(idx(p, q, r + 1))
                = result.col(idx(p, q, r)) * (x.col(2) * ar + br)
                  - result.col(idx(p, q, r - 1)) * cr;
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
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
tabulate_polyset_pyramid(
    int n,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        pts)
{
  assert(pts.cols() == 3);

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x
      = pts * 2.0 - 1.0;

  const int m = (n + 1) * (n + 2) * (2 * n + 3) / 6;
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(
      pts.rows(), m);

  //  const auto f1x = (1.0 + x.col(0) * 2.0 + x.col(2)) * 0.5;
  //  const auto f1y = (1.0 + x.col(1) * 2.0 + x.col(2)) * 0.5;
  const auto f2 = (1.0 - x.col(2)).square() * 0.25;

  // Indexing for pyramidal basis functions
  auto pyr_idx = [&n, &m](int p, int q, int r) {
    const int rv = (n - r);
    const int r0 = rv * (rv + 1) * (2 * rv + 1) / 6;
    const int idx = r0 + p * (rv + 1) + q;
    assert(idx < m);
    return idx;
  };

  result.col(pyr_idx(0, 0, 0)).fill(1.0);
  if (n > 0)
  {
    result.col(pyr_idx(1, 0, 0)) = (1.0 + x.col(0) * 2.0 + x.col(2)) * 0.5;
    result.col(pyr_idx(0, 1, 0)) = (1.0 + x.col(1) * 2.0 + x.col(2)) * 0.5;
  }
  else
    return result;

  // r = 0
  for (int p = 1; p < n; ++p)
  {
    const double a = static_cast<double>(p) / static_cast<double>(p + 1);
    result.col(pyr_idx(p + 1, 0, 0))
        = result.col(pyr_idx(1, 0, 0)) * result.col(pyr_idx(p, 0, 0))
              * (a + 1.0)
          - f2 * result.col(pyr_idx(p - 1, 0, 0)) * a;
  }

  for (int q = 1; q < n; ++q)
  {
    const double a = static_cast<double>(q) / static_cast<double>(q + 1);
    result.col(pyr_idx(0, q + 1, 0))
        = result.col(pyr_idx(0, 1, 0)) * result.col(pyr_idx(0, q, 0))
              * (a + 1.0)
          - f2 * result.col(pyr_idx(0, q - 1, 0)) * a;
  }

  for (int p = 1; p < n + 1; ++p)
    for (int q = 1; q < n + 1; ++q)
    {
      result.col(pyr_idx(p, q, 0))
          = result.col(pyr_idx(p, 0, 0)) * result.col(pyr_idx(0, q, 0));
    }

  // Extend into r > 0
  for (int p = 0; p < n; ++p)
    for (int q = 0; q < n; ++q)
    {
      result.col(pyr_idx(p, q, 1))
          = result.col(pyr_idx(p, q, 0))
            * ((1.0 + p + q) + x.col(2) * (2.0 + p + q));
    }

  for (int r = 1; r < n + 1; ++r)
    for (int p = 0; p < n - r; ++p)
      for (int q = 0; q < n - r; ++q)
      {
        auto [ar, br, cr] = jrc(2 * p + 2 * q + 2, r);
        result.col(pyr_idx(p, q, r + 1))
            = result.col(pyr_idx(p, q, r)) * (x.col(2) * ar + br)
              - result.col(pyr_idx(p, q, r - 1)) * cr;
      }

  for (int r = 0; r < n + 1; ++r)
    for (int p = 0; p < n - r + 1; ++p)
      for (int q = 0; q < n - r + 1; ++q)
      {
        result.col(pyr_idx(p, q, r))
            *= sqrt((q + 0.5) * (p + 0.5) * (p + q + r + 1.5));
      }

  return result;
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
  const int md = (n + 1) * (n + 2) / 2;

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
      std::cout << "(" << kx << "," << ky << ")\n";
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
        std::cout << "(" << kx << "," << ky << ", " << kz << ")\n";
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
        std::cout << "(" << kx << "," << ky << ", " << kz << ")\n";
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
  if (celltype == Cell::Type::interval)
    return tabulate_polyset_line_derivs(n, 0, pts)[0];
  else if (celltype == Cell::Type::triangle)
    return tabulate_polyset_triangle_derivs(n, 0, pts)[0];
  else if (celltype == Cell::Type::tetrahedron)
    return tabulate_polyset_tetrahedron_derivs(n, 0, pts)[0];
  else if (celltype == Cell::Type::quadrilateral)
    return tabulate_polyset_quad_derivs(n, 0, pts)[0];
  else if (celltype == Cell::Type::hexahedron)
    return tabulate_polyset_hex_derivs(n, 0, pts)[0];
  else if (celltype == Cell::Type::prism)
    return tabulate_polyset_prism_derivs(n, 0, pts)[0];
  else if (celltype == Cell::Type::pyramid)
    return tabulate_polyset_pyramid(n, pts);

  throw std::runtime_error("Polynomial set: Unsupported cell type");
}
//-----------------------------------------------------------------------------
std::vector<
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
PolynomialSet::tabulate_polynomial_set_deriv(
    Cell::Type celltype, int n, int nderiv,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        pts)
{
  if (celltype == Cell::Type::interval)
    return tabulate_polyset_line_derivs(n, nderiv, pts);
  else if (celltype == Cell::Type::triangle)
    return tabulate_polyset_triangle_derivs(n, nderiv, pts);
  else if (celltype == Cell::Type::tetrahedron)
    return tabulate_polyset_tetrahedron_derivs(n, nderiv, pts);
  else if (celltype == Cell::Type::quadrilateral)
    return tabulate_polyset_quad_derivs(n, nderiv, pts);
  else if (celltype == Cell::Type::hexahedron)
    return tabulate_polyset_hex_derivs(n, nderiv, pts);

  throw std::runtime_error("Polynomial set: Unsupported cell type");
}
