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
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
tabulate_polyset_line(int n,
                      const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                         Eigen::RowMajor>& pts)
{
  assert(pts.cols() == 1);

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x
      = pts * 2.0 - 1.0;

  const int m = (n + 1);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(
      pts.rows(), m);

  result.col(0).fill(1.0);
  if (n > 0)
    result.col(1) = x;

  for (int p = 2; p < n + 1; ++p)
  {
    double a = 1.0 - 1.0 / static_cast<double>(p);
    result.col(p) = x * result.col(p - 1) * (a + 1.0) - result.col(p - 2) * a;
  }

  for (int p = 0; p < n + 1; ++p)
    result.col(p) *= sqrt(p + 0.5);

  return result;
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
tabulate_polyset_deriv_line(
    int n,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        pts)
{
  assert(pts.cols() == 1);

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x
      = pts * 2.0 - 1.0;

  const int m = (n + 1);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> P(
      pts.rows(), m);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Pd(
      pts.rows(), m);

  P.col(0).fill(1.0);
  if (n > 0)
    P.col(1) = x;

  for (int p = 2; p < n + 1; ++p)
  {
    double a = 1.0 - 1.0 / static_cast<double>(p);
    P.col(p) = x * P.col(p - 1) * (a + 1.0) - P.col(p - 2) * a;
  }

  Pd.col(0).fill(0.0);
  if (n > 0)
    Pd.col(1).fill(2.0);
  for (int p = 2; p < n + 1; ++p)
  {
    double a = 1.0 - 1.0 / static_cast<double>(p);
    Pd.col(p) = (2.0 * P.col(p - 1) + x * Pd.col(p - 1)) * (a + 1.0)
                - Pd.col(p - 2) * a;
  }

  for (int p = 0; p < n + 1; ++p)
    Pd.col(p) *= sqrt(p + 0.5);

  return Pd;
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
tabulate_polyset_triangle(
    int n,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        pts)
{
  assert(pts.cols() == 2);

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x
      = pts * 2.0 - 1.0;

  const int m = (n + 1) * (n + 2) / 2;
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(
      pts.rows(), m);

  result.col(0).fill(1.0);
  if (n > 0)
    result.col(1) = x.col(0) + 0.5 * x.col(1) + 0.5;
  else
    return result;

  const auto f3 = (1.0 - x.col(1)) * (1.0 - x.col(1)) * 0.25;

  for (int p = 1; p < n; ++p)
  {
    double a = static_cast<double>(2 * p + 1) / static_cast<double>(p + 1);
    double b = static_cast<double>(p) / static_cast<double>(p + 1);
    result.col(idx(p + 1, 0)) = result.col(1) * result.col(idx(p, 0)) * a
                                - f3 * result.col(idx(p - 1, 0)) * b;
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

  for (int p = 0; p < n + 1; ++p)
    for (int q = 0; q < n - p + 1; ++q)
      result.col(idx(p, q)) *= sqrt((p + 0.5) * (p + q + 1));

  return result;
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
tabulate_polyset_tetrahedron(
    int n,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        pts)
{
  assert(pts.cols() == 3);

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> x
      = pts * 2.0 - 1.0;

  const int m = (n + 1) * (n + 2) * (n + 3) / 6;
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(
      pts.rows(), m);

  result.col(0).fill(1.0);
  if (n > 0)
    result.col(1) = x.col(0) + 0.5 * (x.col(1) + x.col(2)) + 1.0;
  else
    return result;

  const auto f2 = (x.col(1) + x.col(2)).square() * 0.25;
  const auto f3 = (1.0 + x.col(1) * 2.0 + x.col(2)) * 0.5;
  const auto f4 = (1.0 - x.col(2)) * 0.5;
  const auto f5 = f4 * f4;

  for (int p = 1; p < n; ++p)
  {
    double a = static_cast<double>(p) / static_cast<double>(p + 1);
    result.col(idx(p + 1, 0, 0))
        = result.col(1) * result.col(idx(p, 0, 0)) * (a + 1.0)
          - f2 * result.col(idx(p - 1, 0, 0)) * a;
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

  for (int p = 0; p < n + 1; ++p)
    for (int q = 0; q < n - p + 1; ++q)
      for (int r = 0; r < n - p - q + 1; ++r)
        result.col(idx(p, q, r))
            *= sqrt((p + 0.5) * (p + q + 1.0) * (p + q + r + 1.5));

  return result;
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
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
tabulate_polyset_quad(int n,
                      const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                         Eigen::RowMajor>& pts)
{
  assert(pts.cols() == 2);
  const int m = (n + 1) * (n + 1);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(
      pts.rows(), m);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> px
      = tabulate_polyset_line(n, pts.col(0));
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> py
      = tabulate_polyset_line(n, pts.col(1));

  int c = 0;
  for (int i = 0; i < px.cols(); ++i)
    for (int j = 0; j < py.cols(); ++j)
      result.col(c++) = px.col(i) * py.col(j);

  return result;
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
tabulate_polyset_hex(int n,
                     const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>& pts)
{
  assert(pts.cols() == 3);
  const int m = (n + 1) * (n + 1) * (n + 1);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(
      pts.rows(), m);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> px
      = tabulate_polyset_line(n, pts.col(0));
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> py
      = tabulate_polyset_line(n, pts.col(1));
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pz
      = tabulate_polyset_line(n, pts.col(2));

  int c = 0;
  for (int i = 0; i < px.cols(); ++i)
    for (int j = 0; j < py.cols(); ++j)
      for (int k = 0; k < pz.cols(); ++k)
        result.col(c++) = px.col(i) * py.col(j) * pz.col(k);

  return result;
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
tabulate_polyset_prism(int n,
                       const Eigen::Array<double, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>& pts)
{
  assert(pts.cols() == 3);
  const int m = (n + 1) * (n + 1) * (n + 2) / 2;
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(
      pts.rows(), m);
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pxy
      = tabulate_polyset_triangle(n, pts.leftCols(2));
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pz
      = tabulate_polyset_line(n, pts.col(2));

  int c = 0;
  for (int i = 0; i < pxy.cols(); ++i)
    for (int k = 0; k < pz.cols(); ++k)
      result.col(c++) = pxy.col(i) * pz.col(k);

  return result;
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
    return tabulate_polyset_line(n, pts);
  else if (celltype == Cell::Type::triangle)
    return tabulate_polyset_triangle(n, pts);
  else if (celltype == Cell::Type::tetrahedron)
    return tabulate_polyset_tetrahedron(n, pts);
  else if (celltype == Cell::Type::quadrilateral)
    return tabulate_polyset_quad(n, pts);
  else if (celltype == Cell::Type::hexahedron)
    return tabulate_polyset_hex(n, pts);
  else if (celltype == Cell::Type::prism)
    return tabulate_polyset_prism(n, pts);
  else if (celltype == Cell::Type::pyramid)
    return tabulate_polyset_pyramid(n, pts);

  throw std::runtime_error("Polynomial set: Unsupported cell type");
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
PolynomialSet::tabulate_polynomial_set_deriv(
    Cell::Type celltype, int n,
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        pts)
{
  if (celltype == Cell::Type::interval)
    return tabulate_polyset_deriv_line(n, pts);

  throw std::runtime_error("Polynomial set: Unsupported cell type");
}
