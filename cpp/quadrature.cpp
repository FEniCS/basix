// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "quadrature.h"
#include <cmath>
#include <iostream>
#include <vector>

using namespace libtab;

namespace
{
//-----------------------------------------------------------
std::tuple<Eigen::ArrayXd, Eigen::ArrayXd> rec_jacobi(int N, double a, double b)
{
  // Generate the recursion coefficients alpha_k, beta_k

  // P_{k+1}(x) = (x-alpha_k)*P_{k}(x) - beta_k P_{k-1}(x)

  // for the Jacobi polynomials which are orthogonal on [-1,1]
  // with respect to the weight w(x)=[(1-x)^a]*[(1+x)^b]

  // Inputs:
  // N - polynomial order
  // a - weight parameter
  // b - weight parameter

  // Outputs:
  // alpha - recursion coefficients
  // beta - recursion coefficients

  // Adapted from the MATLAB code by Dirk Laurie and Walter Gautschi
  // http://www.cs.purdue.edu/archives/2002/wxg/codes/r_jacobi.m

  double nu = (b - a) / (a + b + 2.0);
  double mu = pow(2.0, (a + b + 1)) * tgamma(a + 1.0) * tgamma(b + 1.0)
              / tgamma(a + b + 2.0);

  Eigen::ArrayXd alpha(N), beta(N);

  alpha[0] = nu;
  beta[0] = mu;

  Eigen::ArrayXd n = Eigen::ArrayXd::LinSpaced(N - 1, 1.0, N - 1);
  Eigen::ArrayXd nab = 2.0 * n + a + b;
  alpha.tail(N - 1) = (b * b - a * a) / (nab * (nab + 2.0));
  beta.tail(N - 1) = 4 * (n + a) * (n + b) * n * (n + a + b)
                     / (nab * nab * (nab + 1.0) * (nab - 1.0));

  return {alpha, beta};
}
//-----------------------------------------------------------------------------
std::tuple<Eigen::ArrayXd, Eigen::ArrayXd> gauss(const Eigen::ArrayXd& alpha,
                                                 const Eigen::ArrayXd& beta)
{
  // Compute the Gauss nodes and weights from the recursion
  // coefficients associated with a set of orthogonal polynomials
  //
  //  Inputs:
  //  alpha - recursion coefficients
  //  beta - recursion coefficients
  //
  // Outputs:
  // x - quadrature nodes
  // w - quadrature weights
  //
  // Adapted from the MATLAB code by Walter Gautschi
  // http://www.cs.purdue.edu/archives/2002/wxg/codes/gauss.m

  Eigen::MatrixXd A = alpha.matrix().asDiagonal();
  int nb = beta.rows();
  assert(nb == A.cols());
  A.bottomLeftCorner(nb - 1, nb - 1)
      += beta.cwiseSqrt().tail(nb - 1).matrix().asDiagonal();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(
      A, Eigen::DecompositionOptions::ComputeEigenvectors);
  Eigen::ArrayXd x = solver.eigenvalues();
  Eigen::ArrayXd w = beta[0] * solver.eigenvectors().row(0).array().square();
  return {x, w};
}
//-------------------------------------------------------
std::tuple<Eigen::ArrayXd, Eigen::ArrayXd> lobatto(const Eigen::ArrayXd& alpha,
                                                   const Eigen::ArrayXd& beta,
                                                   double xl1, double xl2)
{

  // Compute the Lobatto nodes and weights with the preassigned
  // nodes xl1,xl2
  //
  // Inputs:
  //   alpha - recursion coefficients
  //   beta - recursion coefficients
  //   xl1 - assigned node location
  //   xl2 - assigned node location

  // Outputs:
  // x - quadrature nodes
  // w - quadrature weights

  // Based on the section 7 of the paper
  // "Some modified matrix eigenvalue problems"
  // by Gene Golub, SIAM Review Vol 15, No. 2, April 1973, pp.318--334

  int n = alpha.rows() - 1;

  const Eigen::VectorXd bsqrt = beta.segment(1, n - 1).cwiseSqrt();
  const Eigen::VectorXd a1 = alpha.tail(n) - xl1;
  const Eigen::VectorXd a2 = alpha.tail(n) - xl2;

  Eigen::MatrixXd J = a1.asDiagonal();
  J.topRightCorner(n - 1, n - 1) += bsqrt.asDiagonal();
  J.bottomLeftCorner(n - 1, n - 1) += bsqrt.asDiagonal();

  Eigen::VectorXd en = Eigen::VectorXd::Zero(n);
  en[n - 1] = 1;
  double g1 = J.colPivHouseholderQr().solve(en)[n - 1];
  J.diagonal() = a2;
  double g2 = J.colPivHouseholderQr().solve(en)[n - 1];

  Eigen::ArrayXd alpha_l = alpha;
  alpha_l[n] = (g1 * xl2 - g2 * xl1) / (g1 - g2);
  Eigen::ArrayXd beta_l = beta;
  beta_l[n] = (xl2 - xl1) / (g1 - g2);
  auto [x, w] = gauss(alpha_l, beta_l);
  return {x, w};
}
}; // namespace
//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
quadrature::compute_jacobi_deriv(double a, int n, int nderiv,
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
      const double a1 = 2 * k * (k + a) * (2 * k + a - 2);
      const double a2 = (2 * k + a - 1) * (a * a) / a1;
      const double a3 = (2 * k + a - 1) * (2 * k + a) / (2 * k * (k + a));
      const double a4 = 2 * (k + a - 1) * (k - 1) * (2 * k + a) / a1;
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
Eigen::ArrayXd quadrature::compute_gauss_jacobi_points(double a, int m)
{
  /// Computes the m roots of \f$P_{m}^{a,0}\f$ on [-1,1] by Newton's method.
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
      Eigen::ArrayXd f = quadrature::compute_jacobi_deriv(a, m, 1, x.row(k));
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
quadrature::compute_gauss_jacobi_rule(double a, int m)
{
  /// @note Computes on [-1, 1]
  const Eigen::ArrayXd pts = quadrature::compute_gauss_jacobi_points(a, m);
  const Eigen::ArrayXd Jd
      = quadrature::compute_jacobi_deriv(a, m, 1, pts).row(1);

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
std::pair<Eigen::ArrayXd, Eigen::ArrayXd>
quadrature::make_quadrature_line(int m)
{
  auto [ptx, wx] = quadrature::compute_gauss_jacobi_rule(0.0, m);
  Eigen::ArrayXd pts = 0.5 * (ptx + 1.0);
  Eigen::ArrayXd wts = wx * 0.5;
  return {pts, wts};
}
//-----------------------------------------------------------------------------
std::pair<Eigen::Array<double, Eigen::Dynamic, 2, Eigen::RowMajor>,
          Eigen::ArrayXd>
quadrature::make_quadrature_triangle_collapsed(int m)
{
  auto [ptx, wx] = quadrature::compute_gauss_jacobi_rule(0.0, m);
  auto [pty, wy] = quadrature::compute_gauss_jacobi_rule(1.0, m);

  Eigen::Array<double, Eigen::Dynamic, 2, Eigen::RowMajor> pts(m * m, 2);
  Eigen::ArrayXd wts(m * m);

  int c = 0;
  for (int i = 0; i < m; ++i)
  {
    for (int j = 0; j < m; ++j)
    {
      pts(c, 0) = 0.25 * (1.0 + ptx[i]) * (1.0 - pty[j]);
      pts(c, 1) = 0.5 * (1.0 + pty[j]);
      wts[c] = wx[i] * wy[j] * 0.125;
      ++c;
    }
  }

  return {pts, wts};
}
//-----------------------------------------------------------------------------
std::pair<Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>,
          Eigen::ArrayXd>
quadrature::make_quadrature_tetrahedron_collapsed(int m)
{
  auto [ptx, wx] = quadrature::compute_gauss_jacobi_rule(0.0, m);
  auto [pty, wy] = quadrature::compute_gauss_jacobi_rule(1.0, m);
  auto [ptz, wz] = quadrature::compute_gauss_jacobi_rule(2.0, m);

  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> pts(m * m * m, 3);
  Eigen::ArrayXd wts(m * m * m);

  int c = 0;
  for (int i = 0; i < m; ++i)
  {
    for (int j = 0; j < m; ++j)
    {
      for (int k = 0; k < m; ++k)
      {
        pts(c, 0) = 0.125 * (1.0 + ptx[i]) * (1.0 - pty[j]) * (1.0 - ptz[k]);
        pts(c, 1) = 0.25 * (1. + pty[j]) * (1. - ptz[k]);
        pts(c, 2) = 0.5 * (1.0 + ptz[k]);
        wts[c] = wx[i] * wy[j] * wz[k] * 0.125 * 0.125;
        ++c;
      }
    }
  }

  return {pts, wts};
}
//-----------------------------------------------------------------------------
std::pair<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
          Eigen::ArrayXd>
quadrature::make_quadrature(int dim, int m)
{
  if (dim == 1)
    return quadrature::make_quadrature_line(m);
  else if (dim == 2)
    return quadrature::make_quadrature_triangle_collapsed(m);
  else
    return quadrature::make_quadrature_tetrahedron_collapsed(m);
}
//-----------------------------------------------------------------------------
std::pair<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
          Eigen::ArrayXd>
quadrature::make_quadrature(
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
    std::tie(Qpts, Qwts) = quadrature::make_quadrature_line(m);
    scale = bvec.norm();
  }
  else if (dim == 2)
  {
    std::tie(Qpts, Qwts) = quadrature::make_quadrature_triangle_collapsed(m);
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
    std::tie(Qpts, Qwts) = quadrature::make_quadrature_tetrahedron_collapsed(m);
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
//-----------------------------------------------------------------------------
std::tuple<Eigen::ArrayXd, Eigen::ArrayXd>
quadrature::gauss_lobatto_legendre_line_rule(int m)
{
  // Implement the Gauss-Lobatto-Legendre quadrature rules on the interval
  // using Greg von Winckel's implementation. This facilitates implementing
  // spectral elements.
  // The quadrature rule uses m points for a degree of precision of 2m-3.

  if (m < 2)
    throw std::runtime_error(
        "Gauss-Labotto-Legendre quadrature invalid for fewer than 2 points");

  // Calculate the recursion coefficients
  auto [alpha, beta] = rec_jacobi(m, 0, 0);
  // Compute Lobatto nodes and weights
  auto [xs_ref, ws_ref] = lobatto(alpha, beta, -1.0, 1.0);

  // TODO: scaling

  return {xs_ref, ws_ref};
}
