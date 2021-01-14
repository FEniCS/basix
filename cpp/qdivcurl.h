// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "finite-element.h"

namespace basix
{
/// Create Qdiv element
/// @param celltype
/// @param degree
/// @param name
FiniteElement create_qdiv(cell::type celltype, int degree,
                          const std::string& = std::string());

/// Create Qcurl element
/// @param celltype
/// @param degree
/// @param name
FiniteElement create_qcurl(cell::type celltype, int degree,
                           const std::string& = std::string());

namespace dofperms
{
/// Reflect the DOFs of a Qdiv space on a quadrilateral
/// @param degree The degree of the Qdiv space
/// @return A reordering of DOFs of a Qdiv space of the given degree
Eigen::MatrixXd quadrilateral_qdiv_reflection(int degree);

/// Rotate the DOFs of a Qdiv space on a quadrilateral
/// @param degree The number of DOFs along one side of the quadrilateral
/// @return A reordering of DOFs of a Qdiv space of the given degree
Eigen::MatrixXd quadrilateral_qdiv_rotation(int degree);

} // namespace dofperms

} // namespace basix
