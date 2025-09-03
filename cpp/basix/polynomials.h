// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "mdspan.hpp"
#include "types.h"
#include <array>
#include <concepts>
#include <utility>
#include <vector>

/// Polynomials
namespace basix::polynomials
{
/// @brief Polynomial types that can be created.
enum class type
{
  /// Legendre polynomials: polynomials that span the full space on a cell
  legendre = 0,
  /// Lagrange polynomials: polynomials that span the Lagrange space on a cell.
  /// Note that these will be equal to the Legendre polynomials on all cells
  /// except pyramids
  lagrange = 1,
  /// Bernstein polynomials
  bernstein = 2,
};

/// @brief Tabulate a set of polynomials.
/// @param[in] polytype Polynomial type
/// @param[in] celltype Cell type
/// @param[in] d Polynomial degree
/// @param[in] x Points at which to evaluate the basis. The shape is
/// (number of points, geometric dimension).
/// @return Polynomial sets, for each derivative, tabulated at points.
/// The shape is `(basis index, number of points)`.
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
tabulate(polynomials::type polytype, cell::type celltype, int d,
         md::mdspan<const T, md::dextents<std::size_t, 2>> x);

/// @brief Dimension of a polynomial space.
/// @param[in] polytype Polynomial type
/// @param[in] cell Cell type
/// @param[in] d Polynomial degree
/// @return The number terms in the basis spanning a space of
/// polynomial degree `d`.
int dim(polynomials::type polytype, cell::type cell, int d);

} // namespace basix::polynomials
