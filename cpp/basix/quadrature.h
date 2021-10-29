// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include <utility>
#include <vector>
#include <xtensor/xtensor.hpp>
#include <xtl/xspan.hpp>

/// Quadrature rules
namespace basix::quadrature
{

/// Quadrature type
enum class type
{
  Default = 0,
  GaussJacobi = 1,
  GLL = 2,
  XiaoGimbutas = 3,

  ZienkiewiczTaylor = 20,
  Keast = 21,
  StrangFix = 22,
};

/// Make a quadrature rule on a reference cell
/// @param[in] rule Type of quadrature rule (or use quadrature::Default)
/// @param[in] celltype The cell type
/// @param[in] m Maximum degree of polynomial that this quadrature rule
/// will integrate exactly
/// @return List of points and list of weights. The number of points
/// arrays has shape (num points, gdim)
std::pair<xt::xarray<double>, std::vector<double>>
make_quadrature(const quadrature::type rule, cell::type celltype, int m);

/// Make a default quadrature rule on reference cell
/// @param[in] celltype The cell type
/// @param[in] m Maximum degree of polynomial that this quadrature rule
/// will integrate exactly
/// @return List of points and list of weights. The number of points
/// arrays has shape (num points, gdim)
std::pair<xt::xarray<double>, std::vector<double>>
make_quadrature(cell::type celltype, int m);

/// Get the default quadrature type for the given cell and order
/// @param[in] celltype The cell type
/// @param[in] m Maximum degree of polynomial that this quadrature rule
/// will integrate exactly
/// @return The quadrature type that will be used by default
quadrature::type get_default_rule(cell::type celltype, int m);

/// Get Gauss-Lobatto-Legendre (GLL) points on the interval [0, 1].
/// @param[in] m The number of points
/// @raturn An array of GLL points
xt::xtensor<double, 1> get_gll_points(int m);

/// Get Gauss-Legendre (GL) points on the interval [0, 1].
/// @param[in] m The number of points
/// @raturn An array of GL points
xt::xtensor<double, 1> get_gl_points(int m);

} // namespace basix::quadrature
