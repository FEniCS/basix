// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "maps.h"
#include <vector>

/// Functions to transform DOFs in high degree Lagrange spaces

/// The functions in this namespace calculate the permutations that can be used
/// to rotate and reflect DOF points in Lagrange spaces.
namespace basix::doftransforms
{

/// Compute the entity DOF transformations for an element
/// @param[in] cell_type The cell type
/// @param[in] x Interpolation points for the element
/// @param[in] M Interpolation matrix fot the element
/// @param[in] coeffs The coefficients that define the basis functions of the
/// element in terms of the orthonormal basis
/// @param[in] degree The degree of the element
/// @param[in] vs The value size of the element
/// @param[in] map_type The map type used by the element
std::map<cell::type, xt::xtensor<double, 3>> compute_entity_transformations(
    cell::type cell_type,
    const std::array<std::vector<xt::xtensor<double, 2>>, 4>& x,
    const std::array<std::vector<xt::xtensor<double, 3>>, 4>& M,
    const xt::xtensor<double, 2>& coeffs, const int degree, const int vs,
    maps::type map_type);

} // namespace basix::doftransforms
