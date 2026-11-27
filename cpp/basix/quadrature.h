// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "polyset.h"
#include <array>
#include <concepts>
#include <vector>

/// Quadrature rules
namespace basix::quadrature
{

/// Quadrature type
enum class type
{
  Default = 0,
  gauss_jacobi = 1,
  gll = 2,
  xiao_gimbutas = 3,
  zienkiewicz_taylor = 20,
  keast = 21,
  strang_fix = 22,
};

/// @brief Get the Gauss-Jacobi rule for the interval for integrating
/// f(x) * (1-x)^a on the interval [0, 1].
/// @tparam T The floating point type.
/// @param[in] a The exponent a.
/// @param[in] m The number of points.
/// @return Points and weights of a Gauss-Jacobi rule.
template <std::floating_point T>
std::array<std::vector<T>, 2> gauss_jacobi_rule(T a, int m);

/// @brief Make a quadrature rule on a reference cell.
/// @tparam T The floating point type.
/// @param[in] rule Type of quadrature rule (or use quadrature::Default).
/// @param[in] celltype Cell type.
/// @param[in] polytype Polyset type.
/// @param[in] m Maximum degree of polynomial that this quadrature rule
/// will integrate exactly.
/// @return Points and weights. The points are flattened row-major with
/// shape `(num_points, gdim)`; the weights array has length
/// `num_points`.
template <std::floating_point T>
std::array<std::vector<T>, 2> make_quadrature(const quadrature::type rule,
                                              cell::type celltype,
                                              polyset::type polytype, int m);

/// @brief Get the default quadrature type for the given cell and order.
/// @param[in] celltype Cell type.
/// @param[in] m Maximum degree of polynomial that this quadrature rule
/// will integrate exactly.
/// @return Quadrature type that will be used by default.
quadrature::type get_default_rule(cell::type celltype, int m);

/// @brief Get the quadrature type for integral moments in element dual
/// bases.
///
/// Cells sharing a sub-entity see it in different orientations, so
/// moments on it are only conforming for non-polynomial data if the
/// rule is invariant under the symmetry group of the sub-entity.
/// get_default_rule() does not guarantee this.
///
/// For the cells that occur as sub-entities (interval, triangle,
/// quadrilateral) the rule is invariant under the symmetry group of the
/// cell, except on a triangle for `m > 30`, where no symmetric rule is
/// available. For other cells, get_default_rule() is returned and the
/// rule need not be symmetric.
///
/// @param[in] celltype Cell type.
/// @param[in] m Maximum degree of polynomial that this quadrature rule
/// will integrate exactly.
/// @return Quadrature type.
quadrature::type get_symmetric_rule(cell::type celltype, int m);

/// @brief Get Gauss-Lobatto-Legendre (GLL) points on the interval [0,
/// 1].
/// @tparam T The floating point type.
/// @param[in] m Number of points.
/// @return Array of GLL points.
template <std::floating_point T>
std::vector<T> get_gll_points(int m);

/// @brief Get Gauss-Legendre (GL) points on the interval [0, 1].
/// @tparam T The floating point type.
/// @param[in] m Number of points
/// @return Array of GL points.
template <std::floating_point T>
std::vector<T> get_gl_points(int m);

} // namespace basix::quadrature
