
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
  Eigen::Array<double, 1, 1> r, f, fp;

  for (int k = 0; k < m; ++k)
  {
    // Initial guess
    r[0] = -cos((2.0 * k + 1.0) * M_PI / (2.0 * m));
    if (k > 0)
      r[0] = 0.5 * (r[0] + x[k - 1]);

    int j = 0;
    double delta;
    while (j < max_iter)
    {
      double s = 0;
      for (int i = 0; i < k; ++i)
        s += 1.0 / (r[0] - x[i]);
      f = J.tabulate(r);
      fp = Jd.tabulate(r);
      delta = f[0] / (fp[0] - f[0] * s);
      r[0] -= delta;

      if (fabs(delta) < eps)
        break;
      ++j;
    }
    x[k] = r[0];
  }

  return x;
}
//-----------------------------------------------------------------------------
std::pair<Eigen::ArrayXd, Eigen::ArrayXd> compute_gauss_jacobi_rule(double a,
                                                                    int m)
{
  const Eigen::ArrayXd xs = compute_gauss_jacobi_points(a, m);
  const Polynomial Jd = compute_jacobi(a, m).diff(0);
  Eigen::Array<double, 1, 1> f, x;

  double a1 = pow(2.0, a + 1.0);
  double a3 = tgamma(m + 1.0);

  // factorial(m)
  double a5 = 1.0;
  for (int i = 0; i < m; ++i)
    a5 *= (i + 1);

  double a6 = a1 * a3 / a5;

  Eigen::ArrayXd pts(m);
  Eigen::ArrayXd wts(m);

  for (int i = 0; i < m; ++i)
  {
    x[0] = xs[i];
    f = Jd.tabulate(x);
    pts[i] = x[0];
    wts[i] = a6 / (1.0 - x[0] * x[0]) / (f[0] * f[0]);
  }
  return {pts, wts};
}
