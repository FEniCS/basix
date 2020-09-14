
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
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      triangle = ReferenceSimplex::create_simplex(dim);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      test_pts = ReferenceSimplex::create_lattice(triangle, 50, true);

  Lagrange L(dim, 3);
  auto ltab = L.tabulate_basis(test_pts);

  Nedelec2D N(1);
  auto ntab = N.tabulate_basis(test_pts);

  for (int i = 0; i < test_pts.rows(); ++i)
    std::cout << test_pts.row(i) << " " << ntab.row(i) << "\n";

  return 0;
}
