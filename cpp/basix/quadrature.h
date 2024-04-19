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

/// @brief Make a quadrature rule on a reference cell.
/// @param[in] rule Type of quadrature rule (or use quadrature::Default).
/// @param[in] celltype Cell type.
/// @param[in] polytype Polyset type.
/// @param[in] m Maximum degree of polynomial that this quadrature rule
/// will integrate exactly.
/// @return List of points and list of weights. The number of points
/// arrays has shape `(num points, gdim)`.
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

/// @brief Get Gauss-Lobatto-Legendre (GLL) points on the interval [0,
/// 1].
/// @param[in] m Number of points.
/// @return Array of GLL points.
template <std::floating_point T>
std::vector<T> get_gll_points(int m);

/// @brief Get Gauss-Legendre (GL) points on the interval [0, 1].
/// @param[in] m Number of points
/// @return Array of GL points.
template <std::floating_point T>
std::vector<T> get_gl_points(int m);

} // namespace basix::quadrature
