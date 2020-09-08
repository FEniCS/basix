
#include "lagrange.h"
#include "quadrature.h"
#include "simplex.h"
#include <Eigen/Dense>
#include <iomanip>
#include <iostream>

int main()
{
  int dim = 2;

  Lagrange L(dim, 3);

  ReferenceSimplex triangle(dim);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      test_pts = triangle.lattice(50);
  auto ltab = L.tabulate_basis(test_pts);

  for (int i = 0; i < test_pts.rows(); ++i)
    std::cout << std::setprecision(5) << std::fixed << std::setw(10)
              << test_pts.row(i) << " " << ltab.row(i) << "\n";

  int n = 4;
  double a = 1.0;

  Polynomial J = compute_jacobi(a, n);
  Polynomial Jd = J.diff(0);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pts(5,
                                                                             1);

  pts << -1.0, -0.5, 0.0, 0.5, 1.0;

  std::cout << J.tabulate(pts) << std::endl << std::endl;
  std::cout << Jd.tabulate(pts) << std::endl << std::endl;

  Eigen::ArrayXd p = compute_gauss_jacobi_points(a, n);
  std::cout << p << std::endl << std::endl;

  auto [qpts, qwts] = compute_gauss_jacobi_rule(a, n);

  std::cout << qpts << "\n\n" << qwts << "\n\n";

  return 0;
}
