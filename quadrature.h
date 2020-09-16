// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "polynomial.h"
#include <Eigen/Dense>

#pragma once

/// Evaluate the nth jacobi polynomial with weight parameters (a, 0)
Polynomial compute_jacobi(int a, int n);

// Computes Gauss-Jacobi quadrature points
/// Finds the m roots of P_{m}^{a,0} on [-1,1] by Newton's method.
/// @param a weight in jacobi (b=0)
/// @param m order
/// @return list of points in 1D
Eigen::ArrayXd compute_gauss_jacobi_points(double a, int m);

/// Gauss-Jacobi quadrature rule (points and weights)
std::pair<Eigen::ArrayXd, Eigen::ArrayXd> compute_gauss_jacobi_rule(double a,
                                                                    int m);

/// Computw line quadrature rule on [0, 1]
/// @param m order
/// @returns list of 1D points, list of weights
std::pair<Eigen::ArrayXd, Eigen::ArrayXd> make_quadrature_line(int m);

/// Compute triangle quadrature rule on [0, 1]x[0, 1]
/// @param m order
/// @returns list of 2D points, list of weights
std::pair<Eigen::Array<double, Eigen::Dynamic, 2, Eigen::RowMajor>,
          Eigen::ArrayXd>
make_quadrature_triangle_collapsed(int m);

/// Compute tetrahedrom quadrature rule on [0, 1]x[0, 1]x[0,1]
/// @param m order
/// @returns list of 3D points, list of weights
std::pair<Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>,
          Eigen::ArrayXd>
make_quadrature_tetrahedron_collapsed(int m);

/// Utility for quadrature rule on reference simplex of any dimension
/// @param dim geometric domensiom
/// @param m order
/// @returns list of points, list of weights
std::pair<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
          Eigen::ArrayXd>
make_quadrature(int dim, int m);

/// Scaled quadrature rule on arbitrary simplices
/// @param simplex Set of vertices describing simplex
/// @param m order
/// @returns list of points, list of weights
std::pair<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
          Eigen::ArrayXd>
make_quadrature(const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                   Eigen::RowMajor>& simplex,
                int m);
