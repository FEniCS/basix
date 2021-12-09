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
/// @param[in] polytype Polynomial type
/// @param[in] celltype Cell type
/// @param[in] d Polynomial degree
/// @param[in] x Points at which to evaluate the basis. The shape is
/// (number of points, geometric dimension).
/// @return Polynomial sets, for each derivative, tabulated at points.
/// The shape is `(number of points, basis index)`.
xt::xtensor<double, 2> tabulate(polynomials::type polytype, cell::type celltype,
                                int d, const xt::xarray<double>& x);

/// Dimension of a polynomial space
/// @param[in] cell The cell type
/// @param[in] d The polynomial degree
/// @return The number terms in the basis spanning a space of
/// polynomial degree @p d
int dim(polynomials::type polytype, cell::type cell, int d);

} // namespace basix::polynomials
