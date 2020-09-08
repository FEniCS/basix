
#include "lagrange.h"
#include "simplex.h"
#include <Eigen/Dense>
#include <iomanip>
#include <iostream>

int main()
{
  Lagrange L(2, 3);

  ReferenceSimplex triangle(2);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      test_pts = triangle.lattice(50);
  auto ltab = L.tabulate_basis(test_pts);

  for (int i = 0; i < test_pts.rows(); ++i)
    std::cout << std::setprecision(5) << std::fixed << std::setw(10)
              << test_pts.row(i) << " " << ltab.row(i) << "\n";

  return 0;
}
