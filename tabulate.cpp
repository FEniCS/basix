
#include <Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <vector>

#include "polyn.h"
#include "tabulate.h"

Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
tabulate_line(
    int n,
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pts)
{
  PolyN<1> one = PolyN<1>::one();
  PolyN<1> x = PolyN<1>::x();

  const int m = (n + 1);
  std::vector<PolyN<1>> poly(m);
  poly[0] = one;
  if (n > 0)
    poly[1] = x;

  for (int p = 2; p < n + 1; ++p)
  {
    double a = 1.0 - 1.0 / static_cast<double>(p);
    poly[p] = x * poly[p - 1] * (a + 1.0) - poly[p - 2] * a;
  }

  for (int p = 0; p < n + 1; ++p)
    poly[p] *= sqrt(p + 0.5);

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(
      pts.rows(), poly.size());
  for (int j = 0; j < poly.size(); ++j)
    result.col(j) = poly[j].tabulate(pts);

  return result;
}

Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
tabulate_triangle(int n,
                  Eigen::Array<double, Eigen::Dynamic, 2, Eigen::RowMajor> pts)
{
  PolyN<2> one = PolyN<2>::one();
  PolyN<2> x = PolyN<2>::x();
  PolyN<2> y = PolyN<2>::y();

  const int m = (n + 1) * (n + 2) / 2;
  std::vector<PolyN<2>> poly(m);
  poly[0] = one;
  PolyN<2> f1 = x + (one + y) * 0.5;
  PolyN<2> f2 = (one - y) * 0.5;
  PolyN<2> f3 = f2 * f2;
  if (n > 0)
    poly[1] = f1;

  for (int p = 1; p < n; ++p)
  {
    double a = (2 * p + 1) / static_cast<double>(p + 1);
    double b = p / static_cast<double>(p + 1);
    poly[idx(p + 1, 0)]
        = f1 * poly[idx(p, 0)] * a - f3 * poly[idx(p - 1, 0)] * b;
  }

  for (int p = 0; p < n; ++p)
  {
    poly[idx(p, 1)] = poly[idx(p, 0)] * (one * (0.5 + p) + y * (1.5 + p));
    for (int q = 1; q < n - p; ++q)
    {
      auto [a1, a2, a3] = jrc(2 * p + 1, q);
      poly[idx(p, q + 1)]
          = poly[idx(p, q)] * (y * a1 + one * a2) - poly[idx(p, q - 1)] * a3;
    }
  }

  for (int p = 0; p < n + 1; ++p)
    for (int q = 0; q < n - p + 1; ++q)
      poly[idx(p, q)] *= sqrt((p + 0.5) * (p + q + 1));

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(
      pts.rows(), poly.size());
  for (int j = 0; j < poly.size(); ++j)
    result.col(j) = poly[j].tabulate(pts);

  return result;
}

Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
tabulate_tetrahedron(
    int n, Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> pts)
{
  const PolyN<3> one = PolyN<3>::one();
  const PolyN<3> x = PolyN<3>::x();
  const PolyN<3> y = PolyN<3>::y();
  const PolyN<3> z = PolyN<3>::z();

  const int m = (n + 1) * (n + 2) * (n + 3) / 6;
  std::vector<PolyN<3>> poly(m);
  poly[0] = one;
  const PolyN<3> f1 = one + x + (y + z) * 0.5;
  const PolyN<3> f2 = (y + z) * (y + z) * 0.25;
  const PolyN<3> f3 = (one + y * 2.0 + z) * 0.5;
  const PolyN<3> f4 = (one - z) * 0.5;
  const PolyN<3> f5 = f4 * f4;

  if (n > 0)
    poly[1] = f1;

  for (int p = 1; p < n; ++p)
  {
    double a = static_cast<double>(p) / static_cast<double>(p + 1);
    poly[idx(p + 1, 0, 0)]
      = f1 * poly[idx(p, 0, 0)] * (a + 1.0) - f2 * poly[idx(p - 1, 0, 0)] * a;
  }

  for (int p = 0; p < n; ++p)
  {
    poly[idx(p, 1, 0)] = poly[idx(p, 0, 0)]
                         * ((one + y) * static_cast<double>(p)
                            + (one * 2.0 + y * 3.0 + z) * 0.5);
    for (int q = 1; q < n - p; ++q)
    {
      auto [aq, bq, cq] = jrc(2 * p + 1, q);
      PolyN<3> qmcoeff = f3 * aq + f4 * bq;
      PolyN<3> qm1coeff = f5 * cq;
      poly[idx(p, q + 1, 0)]
          = poly[idx(p, q, 0)] * qmcoeff - poly[idx(p, q - 1, 0)] * qm1coeff;
    }
  }

  for (int p = 0; p < n; ++p)
    for (int q = 0; q < n - p; ++q)
      poly[idx(p, q, 1)]
          = poly[idx(p, q, 0)] * (one * (1.0 + p + q) + z * (2.0 + p + q));

  for (int p = 0; p < n - 1; ++p)
    for (int q = 0; q < n - p - 1; ++q)
      for (int r = 1; r < n - p - q; ++r)
      {
        auto [ar, br, cr] = jrc(2 * p + 2 * q + 2, r);
        poly[idx(p, q, r + 1)] = poly[idx(p, q, r)] * (z * ar + one * br)
                                 - poly[idx(p, q, r - 1)] * cr;
      }

  for (int p = 0; p < n + 1; ++p)
    for (int q = 0; q < n - p + 1; ++q)
      for (int r = 0; r < n - p - q + 1; ++r)
        poly[idx(p, q, r)]
            *= sqrt((p + 0.5) * (p + q + 1.0) * (p + q + r + 1.5));

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(
      pts.rows(), poly.size());
  for (int j = 0; j < poly.size(); ++j)
    result.col(j) = poly[j].tabulate(pts);

  return result;
}
