
#include "lattice.h"
#include "tabulate.h"
#include <Eigen/Dense>
#include <iomanip>
#include <iostream>

int main()
{
  int n = 2;

  // Reference triangle
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> v(3, 2);
  v << -1.0, -1.0, -1.0, 1.0, 1.0, -1.0;
  
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pt = create_lattice(n, v);
  Eigen::MatrixXd r = tabulate_triangle(n, pt).transpose();
  Eigen::MatrixXd w = r.inverse();

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pt2 = create_lattice(50, v);
  Eigen::MatrixXd vals = tabulate_triangle(n, pt2).matrix();

  for (int i = 0; i < pt2.rows(); ++i)
    {
      Eigen::Matrix r = vals.row(i).transpose();
      auto t2 = w * r;
      std::cout << t2.rows() << "x" << t2.cols() << "\n";
						    //    std::cout << std::setprecision(5) << std::fixed << std::setw(10) << pt2.row(i) << " " << w * (vals.row(i).transpose()).transpose() << "\n";
    }

  
  return 0;
}
