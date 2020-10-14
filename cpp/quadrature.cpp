// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "quadrature.h"
#include <cmath>
#include <iostream>
#include <vector>

//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
libtab::compute_jacobi_deriv(double a, int n, int nderiv,
                             const Eigen::ArrayXd& x)
{
  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      J(nderiv + 1);

  for (int i = 0; i < nderiv + 1; ++i)
  {
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& Jd
        = J[i];
    Jd.resize(n + 1, x.rows());

    if (i == 0)
      Jd.row(0).fill(1.0);
    else
      Jd.row(0).setZero();

    if (n > 0)
    {
      if (i == 0)
        Jd.row(1) = (x.transpose() * (a + 2.0) + a) * 0.5;
      else if (i == 1)
        Jd.row(1) = a * 0.5 + 1;
      else
        Jd.row(1).setZero();
    }

    for (int k = 2; k < n + 1; ++k)
    {
      double a1 = 2 * k * (k + a) * (2 * k + a - 2);
      double a2 = (2 * k + a - 1) * (a * a) / a1;
      double a3 = (2 * k + a - 1) * (2 * k + a) / (2 * k * (k + a));
      double a4 = 2 * (k + a - 1) * (k - 1) * (2 * k + a) / a1;
      Jd.row(k)
          = Jd.row(k - 1) * (x.transpose() * a3 + a2) - Jd.row(k - 2) * a4;
      if (i > 0)
        Jd.row(k) += i * a3 * J[i - 1].row(k - 1);
    }
  }

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result(
      nderiv + 1, x.rows());
  for (int i = 0; i < nderiv + 1; ++i)
    result.row(i) = J[i].row(n);

  return result;
}
//-----------------------------------------------------------------------------
Eigen::ArrayXd libtab::compute_gauss_jacobi_points(double a, int m)
{
  /// Computes the m roots of P_{m}^{a,0} on [-1,1] by Newton's method.
  ///    The initial guesses are the Chebyshev points.  Algorithm
  ///    implemented from the pseudocode given by Karniadakis and
  ///    Sherwin

  const double eps = 1.e-8;
  const int max_iter = 100;
  Eigen::ArrayXd x(m);

  for (int k = 0; k < m; ++k)
  {
    // Initial guess
    x[k] = -cos((2.0 * k + 1.0) * M_PI / (2.0 * m));
    if (k > 0)
      x[k] = 0.5 * (x[k] + x[k - 1]);

    int j = 0;
    while (j < max_iter)
    {
      double s = 0;
      for (int i = 0; i < k; ++i)
        s += 1.0 / (x[k] - x[i]);
      Eigen::ArrayXd f = libtab::compute_jacobi_deriv(a, m, 1, x.row(k));
      double delta = f[0] / (f[1] - f[0] * s);
      x[k] -= delta;

      if (fabs(delta) < eps)
        break;
      ++j;
    }
  }

  return x;
}
//-----------------------------------------------------------------------------
std::pair<Eigen::ArrayXd, Eigen::ArrayXd>
libtab::compute_gauss_jacobi_rule(double a, int m)
{
  // Computes on [-1, 1]
  const Eigen::ArrayXd pts = libtab::compute_gauss_jacobi_points(a, m);
  const Eigen::ArrayXd Jd = libtab::compute_jacobi_deriv(a, m, 1, pts).row(1);

  const double a1 = pow(2.0, a + 1.0);
  const double a3 = tgamma(m + 1.0);
  // factorial(m)
  double a5 = 1.0;
  for (int i = 0; i < m; ++i)
    a5 *= (i + 1);
  const double a6 = a1 * a3 / a5;

  Eigen::ArrayXd wts(m);
  for (int i = 0; i < m; ++i)
  {
    const double x = pts[i];
    const double f = Jd[i];
    wts[i] = a6 / (1.0 - x * x) / (f * f);
  }

  return {pts, wts};
}
//-----------------------------------------------------------------------------
std::pair<Eigen::ArrayXd, Eigen::ArrayXd> libtab::make_quadrature_line(int m)
{
  auto [ptx, wx] = libtab::compute_gauss_jacobi_rule(0.0, m);
  Eigen::ArrayXd pts = 0.5 * (ptx + 1.0);
  Eigen::ArrayXd wts = wx * 0.5;
  return {pts, wts};
}
//-----------------------------------------------------------------------------
std::pair<Eigen::Array<double, Eigen::Dynamic, 2, Eigen::RowMajor>,
          Eigen::ArrayXd>
libtab::make_quadrature_triangle_collapsed(int m)
{
  auto [ptx, wx] = libtab::compute_gauss_jacobi_rule(0.0, m);
  auto [pty, wy] = libtab::compute_gauss_jacobi_rule(1.0, m);

  Eigen::Array<double, Eigen::Dynamic, 2, Eigen::RowMajor> pts(m * m, 2);
  Eigen::ArrayXd wts(m * m);

  int idx = 0;
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < m; ++j)
    {
      pts(idx, 0) = 0.25 * (1.0 + ptx[i]) * (1.0 - pty[j]);
      pts(idx, 1) = 0.5 * (1.0 + pty[j]);
      wts[idx] = wx[i] * wy[j] * 0.125;
      ++idx;
    }

  return {pts, wts};
}
//-----------------------------------------------------------------------------
std::pair<Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>,
          Eigen::ArrayXd>
libtab::make_quadrature_tetrahedron_collapsed(int m)
{
  auto [ptx, wx] = libtab::compute_gauss_jacobi_rule(0.0, m);
  auto [pty, wy] = libtab::compute_gauss_jacobi_rule(1.0, m);
  auto [ptz, wz] = libtab::compute_gauss_jacobi_rule(2.0, m);

  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> pts(m * m * m, 3);
  Eigen::ArrayXd wts(m * m * m);

  int idx = 0;
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < m; ++j)
      for (int k = 0; k < m; ++k)
      {
        const double x
            = 0.125 * (1.0 + ptx[i]) * (1.0 - pty[j]) * (1.0 - ptz[k]);
        const double y = 0.25 * (1. + pty[j]) * (1. - ptz[k]);
        const double z = 0.5 * (1.0 + ptz[k]);
        pts.row(idx) << x, y, z;
        wts[idx] = wx[i] * wy[j] * wz[k] * 0.125 * 0.125;
        ++idx;
      }

  return {pts, wts};
}
//-----------------------------------------------------------------------------
std::pair<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
          Eigen::ArrayXd>
libtab::make_quadrature(int dim, int m)
{
  if (dim == 1)
    return libtab::make_quadrature_line(m);
  else if (dim == 2)
    return libtab::make_quadrature_triangle_collapsed(m);
  else
    return libtab::make_quadrature_tetrahedron_collapsed(m);
}
//-----------------------------------------------------------------------------
std::pair<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
          Eigen::ArrayXd>
libtab::make_quadrature(
    const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
        simplex,
    int m)
{
  const int dim = simplex.rows() - 1;
  if (dim < 1 or dim > 3)
    throw std::runtime_error("Unsupported dim");
  if (simplex.cols() < dim)
    throw std::runtime_error("Invalid simplex");

  // Compute edge vectors of simplex
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> bvec(
      dim, simplex.cols());
  for (int i = 0; i < dim; ++i)
    bvec.row(i) = simplex.row(i + 1) - simplex.row(0);

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Qpts;
  Eigen::ArrayXd Qwts;

  double scale = 1.0;
  if (dim == 1)
  {
    std::tie(Qpts, Qwts) = libtab::make_quadrature_line(m);
    scale = bvec.norm();
  }
  else if (dim == 2)
  {
    std::tie(Qpts, Qwts) = libtab::make_quadrature_triangle_collapsed(m);
    if (bvec.cols() == 2)
      scale = bvec.determinant();
    else
    {
      Eigen::Vector3d a = bvec.row(0);
      Eigen::Vector3d b = bvec.row(1);
      scale = a.cross(b).norm();
    }
  }
  else
  {
    std::tie(Qpts, Qwts) = libtab::make_quadrature_tetrahedron_collapsed(m);
    assert(bvec.cols() == 3);
    scale = bvec.determinant();
  }

#ifndef NDEBUG
  std::cout << "vecs = \n[" << bvec << "]\n";
  std::cout << "scale = " << scale << "\n";
#endif

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Qpts_scaled(Qpts.rows(), bvec.cols());
  Eigen::ArrayXd Qwts_scaled = Qwts * scale;

  for (int i = 0; i < Qpts.rows(); ++i)
  {
    Eigen::RowVectorXd s = Qpts.row(i).matrix() * bvec;
    Qpts_scaled.row(i) = simplex.row(0) + s.array();
  }

  return {Qpts_scaled, Qwts_scaled};
}
