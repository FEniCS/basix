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
// Compute polynomial set on 1D line on given axis (dim 1=x, 2=y, 3=z)
std::vector<Polynomial> create_polyset_line(int n, int axis, int dim)
{
  assert(dim > 0 and dim < 4);
  assert(axis < dim);

  // Polynomial representing x, y, or z.
  std::vector<double> w(dim + 1);
  w[axis + 1] = 1.0;
  const Polynomial x0(w, dim);

  const Polynomial one = Polynomial::one();
  const Polynomial x = x0 * 2.0 - one;

  const int m = (n + 1);
  std::vector<Polynomial> poly_set(m);
  poly_set[0] = one;
  if (n > 0)
    poly_set[1] = x;

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
  const Polynomial one = Polynomial::one();
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
  const Polynomial one = Polynomial::one();
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
//-----------------------------------------------------------------------------
std::vector<Polynomial> create_polyset_quad(int n)
{
  const std::vector<Polynomial> px = create_polyset_line(n, 0, 2);
  const std::vector<Polynomial> py = create_polyset_line(n, 1, 2);
  std::vector<Polynomial> pxy;
  for (const Polynomial& p : px)
    for (const Polynomial& q : py)
      pxy.push_back(p * q);

  return pxy;
}
//-----------------------------------------------------------------------------
std::vector<Polynomial> create_polyset_hex(int n)
{
  const std::vector<Polynomial> px = create_polyset_line(n, 0, 3);
  const std::vector<Polynomial> py = create_polyset_line(n, 1, 3);
  const std::vector<Polynomial> pz = create_polyset_line(n, 2, 3);
  std::vector<Polynomial> pxyz;
  for (const Polynomial& p : px)
    for (const Polynomial& q : py)
      for (const Polynomial& r : pz)
        pxyz.push_back(p * q * r);

  return pxyz;
}

} // namespace
//-----------------------------------------------------------------------------
std::vector<Polynomial>
PolynomialSet::compute_polynomial_set(Cell::Type celltype, int n)
{
  if (celltype == Cell::Type::interval)
    return create_polyset_line(n, 0, 1);
  else if (celltype == Cell::Type::triangle)
    return create_polyset_triangle(n);
  else if (celltype == Cell::Type::tetrahedron)
    return create_polyset_tetrahedron(n);
  else if (celltype == Cell::Type::quadrilateral)
    return create_polyset_quad(n);
  else if (celltype == Cell::Type::hexahedron)
    return create_polyset_hex(n);
  else
    throw std::runtime_error("Polynomial set: Unsupported cell type");
}
