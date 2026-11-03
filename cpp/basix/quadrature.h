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
///
/// The `m`-point rule satisfies
/// \f[
///   \int_0^1 f(x) (1 - x)^a \, \mathrm{d} x
///     = \sum_{i = 0}^{m - 1} w_i f(x_i)
/// \f]
/// for every polynomial \f$f\f$ of degree at most \f$2 m - 1\f$,
/// the highest degree attainable with `m` points. The points are the
/// roots of the Jacobi polynomial \f$P_m^{(a, 0)}\f$ mapped to
/// \f$[0, 1]\f$; they lie in the open interval and the weights are
/// positive. Setting `a` to zero gives the Gauss-Legendre rule.
///
/// @tparam T The floating point type.
/// @param[in] a The exponent a, which must be greater than -1.
/// @param[in] m The number of points.
/// @return Points, in increasing order, and weights of a Gauss-Jacobi
/// rule.
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

/// @brief Get Gauss-Lobatto-Legendre (GLL) points on the interval [0,
/// 1].
///
/// These are the points of the `m`-point quadrature rule that has the
/// two endpoints of the interval among its points and is exact for
/// polynomials of degree at most \f$2 m - 3\f$. Prescribing the
/// endpoints costs two degrees relative to the Gauss-Legendre rule on
/// the same number of points, and in return gives a set of points that
/// can serve as the nodes of a nodal basis on a mesh of cells that
/// share their endpoints.
///
/// The two endpoints are returned first, in the order 0, 1, followed
/// by the interior points in increasing order, to match the Basix
/// ordering of degrees of freedom on an interval.
///
/// @tparam T The floating point type.
/// @param[in] m Number of points, at least two.
/// @return Array of GLL points.
template <std::floating_point T>
std::vector<T> get_gll_points(int m);

/// @brief Get Gauss-Legendre (GL) points on the interval [0, 1].
///
/// These are the points of the `m`-point quadrature rule that is exact
/// for polynomials of degree at most \f$2 m - 1\f$, which is the
/// highest degree attainable with `m` points. They are the roots of the
/// Legendre polynomial of degree `m` mapped to \f$[0, 1]\f$, and all
/// lie in the open interval.
///
/// @tparam T The floating point type.
/// @param[in] m Number of points
/// @return Array of GL points, in increasing order.
template <std::floating_point T>
std::vector<T> get_gl_points(int m);

} // namespace basix::quadrature
