
#include "lagrange.h"
#include "nedelec.h"
#include "quadrature.h"
#include "simplex.h"
#include <Eigen/Dense>
#include <iomanip>
#include <iostream>

int main()
{
  int dim = 2;
  ReferenceSimplex triangle(dim);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      test_pts = triangle.lattice(50);

  Lagrange L(dim, 3);
  auto ltab = L.tabulate_basis(test_pts);

  Nedelec2D N(1);
  auto ntab = N.tabulate_basis(test_pts);

  return 0;
}
