// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "span.hpp"
#include <Eigen/Dense>
#include <utility>

/// basix

namespace basix
{
/// Integration using Gauss-Jacobi quadrature on simplices. Other shapes
/// can be obtained by using a product.
/// @todo - pyramid
namespace quadrature
{
/// Evaluate the nth Jacobi polynomial and derivatives with weight parameters
/// (a, 0) at points x
/// @param a Jacobi weight a
/// @param n Order of polynomial
/// @param nderiv Number of derivatives (if zero, just compute polynomial
/// itself)
/// @param x Points at which to evaluate
/// @returns Array of polynomial derivative values (rows) at points (columns)
Eigen::ArrayXXd compute_jacobi_deriv(double a, int n, int nderiv,
                                     const tcb::span<const double>& x);

// Computes Gauss-Jacobi quadrature points
/// Finds the m roots of \f$P_{m}^{a,0}\f$ on [-1,1] by Newton's method.
/// @param a weight in Jacobi (b=0)
/// @param m order
/// @return list of points in 1D
Eigen::ArrayXd compute_gauss_jacobi_points(double a, int m);

/// Gauss-Jacobi quadrature rule (points and weights)
std::pair<Eigen::ArrayXd, Eigen::ArrayXd> compute_gauss_jacobi_rule(double a,
                                                                    int m);

/// Compute line quadrature rule on [0, 1]
/// @param m order
/// @returns list of 1D points, list of weights
std::pair<Eigen::ArrayXd, Eigen::ArrayXd> make_quadrature_line(int m);

/// Compute triangle quadrature rule on [0, 1]x[0, 1]
/// @param m order
/// @returns list of 2D points, list of weights
std::pair<Eigen::ArrayX2d, Eigen::ArrayXd>
make_quadrature_triangle_collapsed(int m);

/// Compute tetrahedron quadrature rule on [0, 1]x[0, 1]x[0, 1]
/// @param m order
/// @returns list of 3D points, list of weights
std::pair<Eigen::ArrayX3d, Eigen::ArrayXd>
make_quadrature_tetrahedron_collapsed(int m);

/// Utility for quadrature rule on reference cell
/// @param rule Name of quadrature rule (or use "default")
/// @param celltype
/// @param m Maximum degree of polynomial that this quadrature rule
///          will integrate exactly
/// @returns list of points, list of weights
std::pair<Eigen::ArrayXXd, Eigen::ArrayXd>
make_quadrature(const std::string& rule, cell::type celltype, int m);

/// Compute GLL quadrature points and weights on the interval [-1, 1]
/// @param m order
/// @return Array of points, array of weights
std::pair<Eigen::ArrayXd, Eigen::ArrayXd>
gauss_lobatto_legendre_line_rule(int m);

} // namespace quadrature
} // namespace basix
