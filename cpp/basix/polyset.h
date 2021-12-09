// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>

/// Polynomial expansion sets
namespace basix::polyset
{
/// Tabulate the orthonormal polynomial basis, and derivatives, at
/// points on the reference cell.
///
/// All derivatives up to the given order are computed. If derivatives
/// are not required, use `n = 0`. For example, order `n = 2` for a 2D
/// cell, will compute the basis \f$\psi, d\psi/dx, d\psi/dy, d^2
/// \psi/dx^2, d^2\psi/dxdy, d^2\psi/dy^2\f$ in that order (0, 0), (1,
/// 0), (0, 1), (2, 0), (1, 1), (0 ,2).
///
/// For an interval cell there are `nderiv + 1` derivatives, for a 2D
/// cell, there are `(nderiv + 1)(nderiv + 2)/2` derivatives, and in 3D,
/// there are `(nderiv + 1)(nderiv + 2)(nderiv + 3)/6`. The ordering is
/// 'triangular' with the lower derivatives appearing first.
///
/// @param[in] celltype Cell type
/// @param[in] d Polynomial degree
/// @param[in] n Maximum derivative order. Use n = 0 for the basis only.
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
xt::xtensor<double, 3> tabulate(cell::type celltype, int d, int n,
                                const xt::xarray<double>& x);

/// Tabulate the orthonormal polynomial basis, and derivatives, at
/// points on the reference cell.
///
/// All derivatives up to the given order are computed. If derivatives
/// are not required, use `n = 0`. For example, order `n = 2` for a 2D
/// cell, will compute the basis \f$\psi, d\psi/dx, d\psi/dy, d^2
/// \psi/dx^2, d^2\psi/dxdy, d^2\psi/dy^2\f$ in that order (0, 0), (1,
/// 0), (0, 1), (2, 0), (1, 1), (0 ,2).
///
/// For an interval cell there are `nderiv + 1` derivatives, for a 2D
/// cell, there are `(nderiv + 1)(nderiv + 2)/2` derivatives, and in 3D,
/// there are `(nderiv + 1)(nderiv + 2)(nderiv + 3)/6`. The ordering is
/// 'triangular' with the lower derivatives appearing first.
///
/// @note This function will be called at runtime when tabulating a finite
/// element, so its performance is critical.
///
/// @param[in,out] P Polynomial sets, for each derivative, tabulated at points.
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
/// @param[in] celltype Cell type
/// @param[in] d Polynomial degree
/// @param[in] n Maximum derivative order. Use n = 0 for the basis only.
/// @param[in] x Points at which to evaluate the basis. The shape is
/// (number of points, geometric dimension).
void tabulate(xt::xtensor<double, 3>& P, cell::type celltype, int d, int n,
              const xt::xarray<double>& x);

/// Dimension of a polynomial space
/// @param[in] cell The cell type
/// @param[in] d The polynomial degree
/// @return The number of terms in the basis spanning a space of
/// polynomial degree @p d
int dim(cell::type cell, int d);

/// Number of derivatives that the orthonormal basis will have on the given cell
/// @param[in] cell The cell type
/// @param[in] d The highest derivative order
/// @return The number of derivatives
/// polynomial degree @p d
int nderivs(cell::type cell, int d);

} // namespace basix::polyset
