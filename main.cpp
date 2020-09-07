
#include "lattice.h"
#include "tabulate.h"
#include <Eigen/Dense>
#include <iomanip>
#include <iostream>

void lagrange_triangle(int n)
{
  // Reference triangle
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> v(3,
                                                                           2);
  v << -1.0, -1.0, 1.0, -1.0, -1.0, 1.0;

  // Evaluate polynomial at nodes, and get inverse
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pt
      = create_lattice(n, v);
  Eigen::MatrixXd r = tabulate_triangle(n, pt);
  Eigen::MatrixXd w = r.inverse();

  // Create a set of points for testing and evaluate at those points
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pt2
      = create_lattice(50, v);
  auto vals = tabulate_triangle(n, pt2);

  for (int i = 0; i < pt2.rows(); ++i)
  {
    Eigen::Matrix<double, 1, Eigen::Dynamic> r = vals.row(i).matrix() * w;
    std::cout << std::setprecision(5) << std::fixed << std::setw(10)
              << pt2.row(i) << " " << r << "\n";
  }
}

int main()
{
  lagrange_triangle(3);

  // Reference tet
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> vtet(
      4, 3);
  vtet << -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0;

  return 0;
}
