// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include <utility>
#include <vector>
#include <xtensor/xtensor.hpp>

namespace basix
{

class FiniteElement;

/// Functions to create integral moment DOFs
namespace moments
{

/// Make interpolation points and weights for simple integral moments
///
/// These will represent the integral of each function in the moment
/// space over each sub entity of the moment space's cell type in a cell
/// with the given type. For example, if the input cell type is a
/// triangle, and the moment space is a P1 space on an edge, this will
/// perform two integrals for each of the 3 edges of the triangle.
///
/// @param moment_space The space to compute the integral moments against
/// @param celltype The cell type of the cell on which the space is
/// being defined
/// @param value_size The value size of the space being defined
/// @param q_deg The quadrature degree used for the integrals
/// @return (interpolation points, interpolation matrix)
std::pair<std::vector<xt::xtensor<double, 2>>,
          std::vector<xt::xtensor<double, 3>>>
make_integral_moments(const FiniteElement& moment_space, cell::type celltype,
                      std::size_t value_size, int q_deg);

/// Make interpolation points and weights for dot product integral
/// moments
///
/// These will represent the integral of each function in the moment
/// space over each sub entity of the moment space's cell type in a cell
/// with the given type. For example, if the input cell type is a
/// triangle and the moment space is a P1 space on an edge, this will
/// perform two integrals for each of the 3 edges of the triangle.
///
/// @todo Clarify what happens value size of the moment space is less
/// than `value_size`.
///
/// @param V The space to compute the integral moments against
/// @param celltype The cell type of the cell on which the space is being
/// defined
/// @param value_size The value size of the space being defined
/// @param q_deg The quadrature degree used for the integrals
/// @return (interpolation points, interpolation matrix)
std::pair<std::vector<xt::xtensor<double, 2>>,
          std::vector<xt::xtensor<double, 3>>>
make_dot_integral_moments(const FiniteElement& V, cell::type celltype,
                          std::size_t value_size, int q_deg);

/// Make interpolation points and weights for tangent integral moments
///
/// These can only be used when the moment space is defined on edges of
/// the cell
///
/// @param V The space to compute the integral moments against
/// @param celltype The cell type of the cell on which the space is
/// being defined
/// @param value_size The value size of the space being defined the
/// space
/// @param q_deg The quadrature degree used for the integrals
/// @return (interpolation points, interpolation matrix)
std::pair<std::vector<xt::xtensor<double, 2>>,
          std::vector<xt::xtensor<double, 3>>>
make_tangent_integral_moments(const FiniteElement& V, cell::type celltype,
                              std::size_t value_size, int q_deg);

/// Compute interpolation points and weights for normal integral moments
///
/// These can only be used when the moment space is defined on facets of
/// the cell
///
/// @param[in] V The space to compute the integral moments against
/// @param[in] celltype The cell type of the cell on which the space is
/// being defined
/// @param[in] value_size The value size of the space being defined
/// @param[in] q_deg The quadrature degree used for the integrals
/// @return (interpolation points, interpolation matrix)
std::pair<std::vector<xt::xtensor<double, 2>>,
          std::vector<xt::xtensor<double, 3>>>
make_normal_integral_moments(const FiniteElement& V, cell::type celltype,
                             std::size_t value_size, int q_deg);

} // namespace moments
} // namespace basix
