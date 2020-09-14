// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "simplex.h"

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
// Compute polynomial set on a line
std::vector<Polynomial> create_polyset_line(int n)
{
  const Polynomial one = Polynomial::one(1);
  const Polynomial x = Polynomial::x(1) * 2.0 - one;

  const int m = (n + 1);
  std::vector<Polynomial> poly_set(m);
  poly_set[0] = one;
  if (n > 0)
    poly_set[1] = x;
  else
    return poly_set;

  for (int p = 2; p < n + 1; ++p)
  {
    double a = 1.0 - 1.0 / static_cast<double>(p);
    poly_set[p] = x * poly_set[p - 1] * (a + 1.0) - poly_set[p - 2] * a;
  }

  for (int p = 0; p < n + 1; ++p)
    poly_set[p] *= sqrt(p + 0.5);

  return poly_set;
}
//-----------------------------------------------------------------------------
std::vector<Polynomial> create_polyset_triangle(int n)
{
  const Polynomial one = Polynomial::one(2);
  const Polynomial x = Polynomial::x(2) * 2.0 - one;
  const Polynomial y = Polynomial::y(2) * 2.0 - one;

  const int m = (n + 1) * (n + 2) / 2;
  std::vector<Polynomial> poly_set(m);
  poly_set[0] = one;
  const Polynomial f1 = x + (one + y) * 0.5;
  const Polynomial f2 = (one - y) * 0.5;
  const Polynomial f3 = f2 * f2;
  if (n > 0)
    poly_set[1] = f1;
  else
    return poly_set;

  for (int p = 1; p < n; ++p)
  {
    double a = (2 * p + 1) / static_cast<double>(p + 1);
    double b = p / static_cast<double>(p + 1);
    poly_set[idx(p + 1, 0)]
        = f1 * poly_set[idx(p, 0)] * a - f3 * poly_set[idx(p - 1, 0)] * b;
  }

  for (int p = 0; p < n; ++p)
  {
    poly_set[idx(p, 1)]
        = poly_set[idx(p, 0)] * (one * (0.5 + p) + y * (1.5 + p));
    for (int q = 1; q < n - p; ++q)
    {
      auto [a1, a2, a3] = jrc(2 * p + 1, q);
      poly_set[idx(p, q + 1)] = poly_set[idx(p, q)] * (y * a1 + one * a2)
                                - poly_set[idx(p, q - 1)] * a3;
    }
  }

  for (int p = 0; p < n + 1; ++p)
    for (int q = 0; q < n - p + 1; ++q)
      poly_set[idx(p, q)] *= sqrt((p + 0.5) * (p + q + 1));

  return poly_set;
}
//-----------------------------------------------------------------------------
std::vector<Polynomial> create_polyset_tetrahedron(int n)
{
  const Polynomial one = Polynomial::one(3);
  const Polynomial x = Polynomial::x(3) * 2.0 - one;
  const Polynomial y = Polynomial::y(3) * 2.0 - one;
  const Polynomial z = Polynomial::z(3) * 2.0 - one;

  const int m = (n + 1) * (n + 2) * (n + 3) / 6;
  std::vector<Polynomial> poly_set(m);
  poly_set[0] = one;
  const Polynomial f1 = one + x + (y + z) * 0.5;
  const Polynomial f2 = (y + z) * (y + z) * 0.25;
  const Polynomial f3 = (one + y * 2.0 + z) * 0.5;
  const Polynomial f4 = (one - z) * 0.5;
  const Polynomial f5 = f4 * f4;

  if (n > 0)
    poly_set[1] = f1;
  else
    return poly_set;

  for (int p = 1; p < n; ++p)
  {
    double a = static_cast<double>(p) / static_cast<double>(p + 1);
    poly_set[idx(p + 1, 0, 0)] = f1 * poly_set[idx(p, 0, 0)] * (a + 1.0)
                                 - f2 * poly_set[idx(p - 1, 0, 0)] * a;
  }

  for (int p = 0; p < n; ++p)
  {
    poly_set[idx(p, 1, 0)] = poly_set[idx(p, 0, 0)]
                             * ((one + y) * static_cast<double>(p)
                                + (one * 2.0 + y * 3.0 + z) * 0.5);
    for (int q = 1; q < n - p; ++q)
    {
      auto [aq, bq, cq] = jrc(2 * p + 1, q);
      const Polynomial qmcoeff = f3 * aq + f4 * bq;
      const Polynomial qm1coeff = f5 * cq;
      poly_set[idx(p, q + 1, 0)] = poly_set[idx(p, q, 0)] * qmcoeff
                                   - poly_set[idx(p, q - 1, 0)] * qm1coeff;
    }
  }

  for (int p = 0; p < n; ++p)
    for (int q = 0; q < n - p; ++q)
      poly_set[idx(p, q, 1)]
          = poly_set[idx(p, q, 0)] * (one * (1.0 + p + q) + z * (2.0 + p + q));

  for (int p = 0; p < n - 1; ++p)
    for (int q = 0; q < n - p - 1; ++q)
      for (int r = 1; r < n - p - q; ++r)
      {
        auto [ar, br, cr] = jrc(2 * p + 2 * q + 2, r);
        poly_set[idx(p, q, r + 1)]
            = poly_set[idx(p, q, r)] * (z * ar + one * br)
              - poly_set[idx(p, q, r - 1)] * cr;
      }

  for (int p = 0; p < n + 1; ++p)
    for (int q = 0; q < n - p + 1; ++q)
      for (int r = 0; r < n - p - q + 1; ++r)
        poly_set[idx(p, q, r)]
            *= sqrt((p + 0.5) * (p + q + 1.0) * (p + q + r + 1.5));

  return poly_set;
}
} // namespace
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
ReferenceSimplex::create_simplex(int dim)
{
  if (dim < 1 or dim > 3)
    throw std::runtime_error("Unsupported dim");

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      ref_geom(dim + 1, dim);
  if (dim == 1)
    ref_geom << 0.0, 1.0;
  else if (dim == 2)
    ref_geom << 0.0, 0.0, 1.0, 0.0, 0.0, 1.0;
  else if (dim == 3)
    ref_geom << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0;
  return ref_geom;
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
ReferenceSimplex::sub(const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                         Eigen::RowMajor>& simplex,
                      int dim, int index)
{
  const int simplex_dim = simplex.rows() - 1;

  if (dim == 0)
  {
    assert(index >= 0 and index < simplex.rows());
    return simplex.row(index);
  }
  else if (dim == simplex_dim)
  {
    assert(index == 0);
    return simplex;
  }

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> entity;

  if (dim == 1 and simplex_dim == 2)
  {
    assert(index >= 0 and index < 3);
    // Edge of triangle
    entity.resize(2, 2);
    entity.row(0) = simplex.row((index + 1) % 3);
    entity.row(1) = simplex.row((index + 2) % 3);
  }
  else if (dim == 2 and simplex_dim == 3)
  {
    assert(index >= 0 and index < 4);
    // Facet of tetrahedron
    entity.resize(3, 3);
    entity.row(0) = simplex.row((index + 1) % 4);
    entity.row(1) = simplex.row((index + 2) % 4);
    entity.row(2) = simplex.row((index + 3) % 4);
  }
  else if (dim == 1 and simplex_dim == 3)
  {
    // Edge of tetrahedron...
    throw std::runtime_error("Fix me");
  }
  return entity;
}
//-----------------------------------------------------------------------------
std::vector<Polynomial> ReferenceSimplex::compute_polynomial_set(
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        simplex,
    int n)

{
  // Could just use dim as an argument, just checking rows to get dimension.
  const int dim = simplex.rows() - 1;
  if (dim < 1 or dim > 3)
    throw std::runtime_error("Unsupported dim");

  if (dim == 1)
    return create_polyset_line(n);
  else if (dim == 2)
    return create_polyset_triangle(n);
  else
    return create_polyset_tetrahedron(n);
}
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
ReferenceSimplex::create_lattice(
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& vertices,
    int n, bool exterior)
{
  int tdim = vertices.rows() - 1;
  int gdim = vertices.cols();
  assert(gdim > 0 and gdim < 4);
  assert(tdim <= gdim);

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> hs(
      tdim, vertices.cols());
  for (int j = 1; j < tdim + 1; ++j)
    hs.row(tdim - j)
        = (vertices.row(j) - vertices.row(0)) / static_cast<double>(n);

  int b = (exterior == false) ? 1 : 0;

  int m = 1;
  for (int j = 0; j < tdim; ++j)
  {
    m *= (n - (tdim + 1) * b + j + 1);
    m /= (j + 1);
  }

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> points(
      m, gdim);

  int c = 0;
  if (tdim == 3)
  {
    for (int i = b; i < n + 1 - b; ++i)
      for (int j = b; j < n + 1 - i - b; ++j)
        for (int k = b; k < n + 1 - i - j - b; ++k)
          points.row(c++)
              = vertices.row(0) + hs.row(2) * k + hs.row(1) * j + hs.row(0) * i;
  }
  else if (tdim == 2)
  {
    for (int i = b; i < n + 1 - b; ++i)
      for (int j = b; j < n + 1 - i - b; ++j)
        points.row(c++) = vertices.row(0) + hs.row(1) * j + hs.row(0) * i;
  }
  else
  {
    for (int i = b; i < n + 1 - b; ++i)
      points.row(c++) = vertices.row(0) + hs.row(0) * i;
  }

  return points;
}
