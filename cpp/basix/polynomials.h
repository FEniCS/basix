// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "mdspan.hpp"
#include <array>
#include <concepts>
#include <utility>
#include <vector>

/// Polynomials
namespace basix::polynomials
{
/// @brief Variants of a Lagrange space that can be created.
enum class type
{
  legendre = 0,
  bernstein = 1,
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
         MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
             const T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 2>>
             x);

/// @brief Dimension of a polynomial space.
/// @param[in] polytype Polynomial type
/// @param[in] cell Cell type
/// @param[in] d Polynomial degree
/// @return The number terms in the basis spanning a space of
/// polynomial degree `d`.
int dim(polynomials::type polytype, cell::type cell, int d);

} // namespace basix::polynomials
