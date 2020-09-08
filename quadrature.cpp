
#include "quadrature.h"
#include "polynomial.h"
#include <vector>

// Evaluates the nth jacobi polynomial with weight parameters a,0
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
