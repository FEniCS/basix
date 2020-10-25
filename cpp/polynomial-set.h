// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "cell.h"
#include <vector>

#pragma once

namespace libtab
{

/// ## Orthonormal polynomial basis on reference cell
/// These are the underlying "expansion sets" for all finite elements, which
/// when multiplied by a set of "coefficients" give the FE basis functions.
///
/// The polynomials (and their derivatives) can be tabulated on unit interval,
/// triangle, tetrahedron, quadrilateral, hexahedron, prism and pyramids.
namespace PolynomialSet
{
/// Basis and derivatives of orthonormal polynomials on reference cell at points
///
/// Compute all derivatives up to given order.
/// If derivatives are not required, use n=0. For example, order n=2 for a 2D
/// cell, will compute the basis \f$N, dN/dx, dN/dy, d^2N/dx^2, d^2N/dxdy,
/// d^2N/dy^2\f$ in that order. For an interval cell there are (nderiv + 1)
/// derivatives, for a 2D cell, there are (nderiv + 1)(nderiv + 2)/2
/// derivatives, and in 3D, there are (nderiv + 1)(nderiv + 2)(nderiv + 3)/6.
/// The ordering is 'triangular' with the lower derivatives appearing first.
///
/// @param celltype Cell type
/// @param n order
/// @param nderiv Maximum derivative order
/// @param pts points
/// @return List of polynomial sets, for each derivative, tabulated at points.
std::vector<
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
tabulate(Cell::Type celltype, int n, int nderiv,
         const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                            Eigen::RowMajor>& pts);

/// Size of set
/// @todo What is n?
int size(Cell::Type celltype, int n);

} // namespace PolynomialSet
} // namespace libtab
