// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include <Eigen/Dense>

namespace basix
{

/// Functions to help with the creation of DOF transformation and direction
/// correction.
namespace doftransforms
{

/// Reflect the DOFs on an interval
/// @param degree The number of DOFs on the interval
/// @return A reordering of the numbers 0 to degree-1 representing the
/// transformation
Eigen::ArrayXi interval_reflection(int degree);

/// Reflect the DOFs on a triangle
/// @param degree The number of DOFs along one side of the triangle
/// @return A reordering of the numbers 0 to (degree)*(degree+1)/2-1
/// representing the transformation
Eigen::ArrayXi triangle_reflection(int degree);

/// Rotate the DOFs on a triangle
/// @param degree The number of DOFs along one side of the triangle
/// @return A reordering of the numbers 0 to (degree)*(degree+1)/2-1
/// representing the transformation
Eigen::ArrayXi triangle_rotation(int degree);

/// Reflect the DOFs on a quadrilateral
/// @param degree The number of DOFs along one side of the quadrilateral
/// @return A reordering of the numbers 0 to degree*degree-1 representing the
/// transformation
Eigen::ArrayXi quadrilateral_reflection(int degree);

/// Rotate the DOFs on a quadrilateral
/// @param degree The number of DOFs along one side of the quadrilateral
/// @return A reordering of the numbers 0 to degree*degree-1 representing the
/// transformation
Eigen::ArrayXi quadrilateral_rotation(int degree);

}; // namespace doftransforms
} // namespace basix
