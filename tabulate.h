
#include <Eigen/Dense>

#pragma once

// Tabulate line basis functions (Legendre polynomials) at given points
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
tabulate_line(
    int n,
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pts);

// Tabulate Dubiner triangle basis functions at given points
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
tabulate_triangle(int n,
                  Eigen::Array<double, Eigen::Dynamic, 2, Eigen::RowMajor> pts);

// Tabulate tetrahedron basis functions at given points
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
tabulate_tetrahedron(
    int n, Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> pts);
