
#include "quadrature.h"
#include "polynomial.h"
#include <cmath>
#include <vector>

// Evaluates the nth jacobi polynomial with weight parameters a,0
// FIXME: similar code in simplex.cpp - remove duplication
Polynomial compute_jacobi(int a, int n)
{
  std::vector<Polynomial> Jn(n + 1);
  const Polynomial x = Polynomial::x(1);
  const Polynomial one = Polynomial::one(1);

  Jn[0] = one;

  if (n > 0)
    Jn[1] = (one * a + x * (a + 2.0)) * 0.5;

  for (int k = 2; k < n + 1; ++k)
  {
    double a1 = 2.0 * k * (k + a) * (2.0 * k + a - 2.0);
    double a2 = (2.0 * k + a - 1.0) * (a * a);
    double a3 = (2.0 * k + a - 2.0) * (2.0 * k + a - 1.0) * (2.0 * k + a);
    double a4 = 2.0 * (k + a - 1.0) * (k - 1.0) * (2.0 * k + a);
    a2 = a2 / a1;
    a3 = a3 / a1;
    a4 = a4 / a1;
    Jn[k] = Jn[k - 1] * (one * a2 + x * a3) - Jn[k - 2] * a4;
  }

  return Jn[n];
}
//-----------------------------------------------------------------------------
Eigen::ArrayXd compute_gauss_jacobi_points(double a, int m)
{
  /// Computes the m roots of P_{m}^{a,0} on [-1,1] by Newton's method.
  ///    The initial guesses are the Chebyshev points.  Algorithm
  ///    implemented from the pseudocode given by Karniadakis and
  ///    Sherwin

  const Polynomial J = compute_jacobi(a, m);
  const Polynomial Jd = J.diff(0);
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
      double f = J.tabulate(x[k]);
      double fp = Jd.tabulate(x[k]);
      double delta = f / (fp - f * s);
      x[k] -= delta;

      if (fabs(delta) < eps)
        break;
      ++j;
    }
  }

  return x;
}
//-----------------------------------------------------------------------------
std::pair<Eigen::ArrayXd, Eigen::ArrayXd> compute_gauss_jacobi_rule(double a,
                                                                    int m)
{
  const Eigen::ArrayXd pts = compute_gauss_jacobi_points(a, m);
  const Polynomial Jd = compute_jacobi(a, m).diff(0);

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
    const double f = Jd.tabulate(x);
    wts[i] = a6 / (1.0 - x * x) / (f * f);
  }

  return {pts, wts};
}
//-----------------------------------------------------------------------------
std::pair<Eigen::ArrayXd, Eigen::ArrayXd> make_quadrature_line(int m)
{
  return compute_gauss_jacobi_rule(0.0, m);
}
//-----------------------------------------------------------------------------
std::pair<Eigen::Array<double, Eigen::Dynamic, 2, Eigen::RowMajor>,
          Eigen::ArrayXd>
make_quadrature_triangle_collapsed(int m)
{
  auto [ptx, wx] = compute_gauss_jacobi_rule(0.0, m);
  auto [pty, wy] = compute_gauss_jacobi_rule(1.0, m);

  Eigen::Array<double, Eigen::Dynamic, 2, Eigen::RowMajor> pts(m * m, 2);
  Eigen::ArrayXd wts(m * m);

  int idx = 0;
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < m; ++j)
    {
      const double x = 0.5 * (1.0 + ptx[i]) * (1.0 - pty[j]) - 1.0;
      const double y = pty[j];
      pts.row(idx) << x, y;
      wts[idx] = wx[i] * wy[j] * 0.5;
      ++idx;
    }

  return {pts, wts};
}
//-----------------------------------------------------------------------------
std::pair<Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>,
          Eigen::ArrayXd>
make_quadrature_tetrahedron_collapsed(int m)
{
  auto [ptx, wx] = compute_gauss_jacobi_rule(0.0, m);
  auto [pty, wy] = compute_gauss_jacobi_rule(1.0, m);
  auto [ptz, wz] = compute_gauss_jacobi_rule(2.0, m);

  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> pts(m * m * m, 3);
  Eigen::ArrayXd wts(m * m * m);

  int idx = 0;
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < m; ++j)
      for (int k = 0; k < m; ++k)
      {
        const double x
            = 0.25 * (1.0 + ptx[i]) * (1.0 - pty[j]) * (1.0 - ptz[k]) - 1.0;
        const double y = 0.5 * (1. + pty[j]) * (1. - ptz[k]) - 1.0;
        const double z = ptz[k];
        pts.row(idx) << x, y, z;
        wts[idx] = wx[i] * wy[j] * wz[k] * 0.125;
        ++idx;
      }

  return {pts, wts};
}
