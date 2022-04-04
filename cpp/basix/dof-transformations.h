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

/// Reflect the DOFs on an interval
/// @param degree The number of DOFs on the interval
/// @return A reordering of the numbers 0 to degree-1 representing the
/// transformation
std::vector<int> interval_reflection(int degree);

/// Reflect the DOFs on a triangle
/// @param degree The number of DOFs along one side of the triangle
/// @return A reordering of the numbers 0 to (degree)*(degree+1)/2-1
/// representing the transformation
std::vector<int> triangle_reflection(int degree);

/// Rotate the DOFs on a triangle
/// @param degree The number of DOFs along one side of the triangle
/// @return A reordering of the numbers 0 to (degree)*(degree+1)/2-1
/// representing the transformation
std::vector<int> triangle_rotation(int degree);

/// Reflect the DOFs on a quadrilateral
/// @param degree The number of DOFs along one side of the quadrilateral
/// @return A reordering of the numbers 0 to degree*degree-1 representing the
/// transformation
std::vector<int> quadrilateral_reflection(int degree);

/// Rotate the DOFs on a quadrilateral
/// @param degree The number of DOFs along one side of the quadrilateral
/// @return A reordering of the numbers 0 to degree*degree-1 representing the
/// transformation
std::vector<int> quadrilateral_rotation(int degree);

} // namespace basix::doftransforms
