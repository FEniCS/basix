
#include <Eigen/Dense>

#pragma once

// Tabulate line basis functions (Legendre polynomials) at given points
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
tabulate_line(int n, const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>& pts);

// Tabulate Dubiner triangle basis functions at given points
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
tabulate_triangle(int n,
                  const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>& pts);

// Tabulate tetrahedron basis functions at given points
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
tabulate_tetrahedron(int n,
                     const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>& pts);
