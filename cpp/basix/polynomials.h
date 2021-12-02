// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>

/// Polynomials
namespace basix::polynomials
{
/// An enum defining the variants of a Lagrange space that can be created
enum class type
{
  legendre = 0,
  chebyshev = 1,
  legendre_bubble = 2,
};

/// Tabulate a set of polynomials.
///
/// @param[in] celltype Cell type
/// @param[in] d Polynomial degree
/// @param[in] x Points at which to evaluate the basis. The shape is
/// (number of points, geometric dimension).
/// @return Polynomial sets, for each derivative, tabulated at points.
/// The shape is `(number of derivatives computed, number of points,
/// basis index)`.
///
/// - The first index is the derivative. The first entry is the basis
/// itself. Derivatives are stored in triangular (2D) or tetrahedral
/// (3D) ordering, e.g. if `(p, q)` denotes `p` order dervative with
/// repsect to `x` and `q` order derivative with respect to `y`, [0] ->
/// (0, 0), [1] -> (1, 0), [2] -> (0, 1), [3] -> (2, 0), [4] -> (1, 1),
/// [5] -> (0, 2), [6] -> (3, 0),...
/// The function basix::indexing::idx maps tuples `(p, q, r)` to the array
/// index.
///
/// - The second index is the point, with index `i` correspondign to the
/// point in row `i` of @p x.
///
/// - The third index is the basis function index.
/// @todo Does the order for the third index need to be documented?
xt::xtensor<double, 2> tabulate(polynomials::type polytype, cell::type celltype,
                                int d, const xt::xarray<double>& x);

/// Dimension of a polynomial space
/// @param[in] cell The cell type
/// @param[in] d The polynomial degree
/// @return The number terms in the basis spanning a space of
/// polynomial degree @p d
int dim(polynomials::type polytype, cell::type cell, int d);

} // namespace basix::polynomials
