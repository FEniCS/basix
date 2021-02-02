// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "finite-element.h"

namespace basix
{
/// Create Raviart-Thomas element
/// @param celltype
/// @param degree
FiniteElement create_rt(cell::type celltype, int degree);

namespace dofperms
{
/// Reflect the DOFs of a RT space on a triangle
/// @param degree The degree of the RT space
/// @return A reordering of DOFs of a RT space of the given degree
Eigen::MatrixXd triangle_rt_reflection(int degree);

/// Rotate the DOFs of a RT space on a triangle
/// @param degree The number of DOFs along one side of the triangle
/// @return A reordering of DOFs of a RT space of the given degree
Eigen::MatrixXd triangle_rt_rotation(int degree);

} // namespace dofperms

} // namespace basix
