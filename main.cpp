
#include "lattice.h"
#include "tabulate.h"
#include <Eigen/Dense>
#include <iomanip>
#include <iostream>

int main()
{

  // Tabulate at some random points

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pts1d(
      3, 1);

  pts1d << -1.0, 0.4, 0.9;

  std::cout << std::setprecision(9) << std::endl
            << tabulate_line(6, pts1d) << std::endl;

  Eigen::Array<double, 3, 2, Eigen::RowMajor> pts2d;

  pts2d.row(0) << -1.0, -1.0;
  pts2d.row(1) << 0.0, 1.0;
  pts2d.row(2) << 1.0, 0.0;

  std::cout << std::setprecision(9) << std::endl
            << tabulate_triangle(3, pts2d) << std::endl;

  Eigen::Array<double, 4, 3, Eigen::RowMajor> pts3d;
  pts3d.row(0) << 0.0, 0.0, 0.0;
  pts3d.row(1) << 0.0, 0.0, 1.0;
  pts3d.row(2) << 0.0, 1.0, 0.0;
  pts3d.row(3) << 1.0, 0.0, 0.0;

  std::cout << tabulate_tetrahedron(3, pts3d) << std::endl;

  auto pt = create_lattice(pts2d, 2);
  std::cout << pt;
  Eigen::MatrixXd w = tabulate_triangle(2, pt);
  std::cout << "\n\n" << w.inverse() << "\n";

  return 0;
}
