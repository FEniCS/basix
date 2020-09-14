// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "polynomial.h"
#include <Eigen/Dense>

#pragma once

// Evaluates the nth jacobi polynomial with weight parameters (a, 0)
Polynomial compute_jacobi(int a, int n);

// points
Eigen::ArrayXd compute_gauss_jacobi_points(double a, int m);

// rule (points and weights)
std::pair<Eigen::ArrayXd, Eigen::ArrayXd> compute_gauss_jacobi_rule(double a,
                                                                    int m);

// line quadrature
std::pair<Eigen::ArrayXd, Eigen::ArrayXd> make_quadrature_line(int m);

// triangle collapsed
std::pair<Eigen::Array<double, Eigen::Dynamic, 2, Eigen::RowMajor>,
          Eigen::ArrayXd>
make_quadrature_triangle_collapsed(int m);

// tet collapsed
std::pair<Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>,
          Eigen::ArrayXd>
make_quadrature_tetrahedron_collapsed(int m);

// Utility for reference simplex of any dimension
std::pair<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
          Eigen::ArrayXd>
  make_quadrature(int dim, int m);

// Scaled quadrature on arbitrary simplices
std::pair<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
          Eigen::ArrayXd>
make_quadrature(const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>& simplex,
                int m);
