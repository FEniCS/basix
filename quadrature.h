// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include <Eigen/Dense>
#include "polynomial.h"

// Evaluates the nth jacobi polynomial with weight parameters (a, 0)
Polynomial compute_jacobi(int a, int n);

// points
Eigen::ArrayXd compute_gauss_jacobi_points(double a, int m);

// rule (points and weights)
std::pair<Eigen::ArrayXd, Eigen::ArrayXd> compute_gauss_jacobi_rule(double a, int m);

// triangle collapsed
std::pair<Eigen::Array<double, Eigen::Dynamic, 2, Eigen::RowMajor>,
          Eigen::ArrayXd>
  make_quadrature_triangle_collapsed(int m);

// tet collapsed
std::pair<Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>,
          Eigen::ArrayXd>
  make_quadrature_tetrahedron_collapsed(int m);
