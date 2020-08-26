
#include <Eigen/Dense>

#pragma once

Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
tabulate_line(
    int n,
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pts);

Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
tabulate_triangle(int n,
                  Eigen::Array<double, Eigen::Dynamic, 2, Eigen::RowMajor> pts);

Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
tabulate_tetrahedron(
    int n, Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> pts);
