// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include <Eigen/Dense>

namespace libtab
{

/// Functions to help with the creation of DOF permutation and direction
/// correction.
namespace dofperms
{

/// Reflect the DOFs on an interval
/// @param degree The number of DOFs on the interval
/// @return A reordering of the numbers 0 to degree-1 representing the
/// permutation
Eigen::Array<int, Eigen::Dynamic, 1> interval_reflection(int degree);

/// Reflect the DOFs on a triangle
/// @param degree The number of DOFs along one side of the triangle
/// @return A reordering of the numbers 0 to degree-1 representing the
/// permutation
Eigen::Array<int, Eigen::Dynamic, 1> triangle_reflection(int degree);

/// Rotate the DOFs on a triangle
/// @param degree The number of DOFs along one side of the triangle
/// @return A reordering of the numbers 0 to degree-1 representing the
/// permutation
Eigen::Array<int, Eigen::Dynamic, 1> triangle_rotation(int degree);

//-----------------------------------------------------------------------------

/// Generate a matrix to correct the direction of tangent vector-values DOFs on
/// an interval when that interval is reflected
/// @param degree The number of DOFs on the interval
/// @return A matrix representing the effect of reversing the edge on the DOF
/// values
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
interval_reflection_tangent_directions(int degree);

/// Generate a matrix to correct the direction of tangent vector-values DOFs on
/// a triangle when that triangle is reflected
/// @param degree The number of DOFs along one side of the triangle
/// @return A matrix representing the effect of reflecting the triangle edge on
/// the DOF values
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
triangle_reflection_tangent_directions(int degree);

/// Generate a matrix to correct the direction of tangent vector-values DOFs on
/// a triangle when that triangle is rotated
/// @param degree The number of DOFs along one side of the triangle
/// @return A matrix representing the effect of rotating the triangle edge on
/// the DOF values
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
triangle_rotation_tangent_directions(int degree);

}; // namespace dofperms
} // namespace libtab
