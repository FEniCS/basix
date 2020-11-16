// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include <Eigen/Dense>
#include <vector>

namespace libtab
{

/// ## Orthonormal polynomial basis on reference cell
/// These are the underlying "expansion sets" for all finite elements, which
/// when multiplied by a set of "coefficients" give the FE basis functions.
///
/// The polynomials (and their derivatives) can be tabulated on unit interval,
/// triangle, tetrahedron, quadrilateral, hexahedron, prism and pyramids.
namespace polyset
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
/// @param degree Polynomial degree
/// @param nd Maximum derivative order. Use nd = 0 for the basis only.
/// @param x Points at which to evaluate the basis. The shape is (number
/// of points, geometric dimension).
/// @return List of polynomial sets, for each derivative, tabulated at
/// points. The first index is the derivative. Higher derivatives are
/// stored in triangular (2D) or tetrahedral (3D) ordering, i.e. for the
/// (x,y) derivatives in 2D: (0,0), (1,0), (0,1), (2,0), (1,1), (0,2),
/// (3,0), ... If a vector result is expected, it will be stacked with
/// all x values, followed by all y-values (and then z, if any). The
/// second index is the point, and the third index is the basis function
/// index.
std::vector<
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
tabulate(cell::Type celltype, int degree, int nd,
         const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                            Eigen::RowMajor>& x);

/// Size of set
/// @todo What is n?
int size(cell::Type celltype, int n);

} // namespace polyset
} // namespace libtab
