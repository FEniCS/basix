// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "span.hpp"
#include <Eigen/Core>
#include <utility>
#include <xtensor/xtensor.hpp>

/// basix

/// Integration using Gauss-Jacobi quadrature on simplices. Other shapes
/// can be obtained by using a product.
/// @todo - pyramid
namespace basix::quadrature
{
/// Evaluate the nth Jacobi polynomial and derivatives with weight
/// parameters (a, 0) at points x
/// @param[in] a Jacobi weight a
/// @param[in] n Order of polynomial
/// @param[in] nderiv Number of derivatives (if zero, just compute
/// polynomial itself)
/// @param[in] x Points at which to evaluate
/// @returns Array of polynomial derivative values (rows) at points
/// (columns)
xt::xtensor<double, 2> compute_jacobi_deriv(double a, std::size_t n,
                                            std::size_t nderiv,
                                            const tcb::span<const double>& x);

// Computes Gauss-Jacobi quadrature points
/// Finds the m roots of \f$P_{m}^{a,0}\f$ on [-1,1] by Newton's method.
/// @param[in] a weight in Jacobi (b=0)
/// @param[in] m order
/// @return list of points in 1D
std::vector<double> compute_gauss_jacobi_points(double a, int m);

/// Gauss-Jacobi quadrature rule (points and weights)
std::pair<xt::xarray<double>, std::vector<double>>
compute_gauss_jacobi_rule(double a, int m);

/// Compute line quadrature rule on [0, 1]
/// @param m order
/// @returns list of points, list of weights
std::pair<xt::xarray<double>, std::vector<double>> make_quadrature_line(int m);

/// Compute triangle quadrature rule on [0, 1]x[0, 1]
/// @param[in] m order
/// @returns list of points, list of weights
std::pair<xt::xarray<double>, std::vector<double>>
make_quadrature_triangle_collapsed(std::size_t m);

/// Compute tetrahedron quadrature rule on [0, 1]x[0, 1]x[0, 1]
/// @param[in] m order
/// @returns List of points, list of weights. The number of points
/// arrays has shape (num points, gdim)
std::pair<xt::xarray<double>, std::vector<double>>
make_quadrature_tetrahedron_collapsed(std::size_t m);

/// Utility for quadrature rule on reference cell
/// @param[in] rule Name of quadrature rule (or use "default")
/// @param[in] celltype
/// @param[in] m Maximum degree of polynomial that this quadrature rule
/// will integrate exactly
/// @returns List of points and list of weights. The number of points
/// arrays has shape (num points, gdim)
std::pair<Eigen::ArrayXXd, Eigen::ArrayXd>
make_quadrature(const std::string& rule, cell::type celltype, int m);

/// Compute GLL line quadrature rule on [0, 1]
/// @param m order
/// @returns list of 1D points, list of weights
std::pair<xt::xarray<double>, std::vector<double>> make_gll_line(int m);

/// GLL quadrature rule (points and weights)
std::pair<xt::xarray<double>, std::vector<double>> compute_gll_rule(int m);

} // namespace basix::quadrature
