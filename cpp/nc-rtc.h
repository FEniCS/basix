// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "finite-element.h"

namespace basix
{
/// Create RTC H(div) element
/// @param celltype
/// @param degree
/// @param name
FiniteElement create_rtc(cell::type celltype, int degree,
                         const std::string& = std::string());

/// Create NC H(curl) element
/// @param celltype
/// @param degree
/// @param name
FiniteElement create_nc(cell::type celltype, int degree,
                        const std::string& = std::string());

namespace dofperms
{
/// Reflect the DOFs of a rtc space on a quadrilateral
/// @param degree The degree of the rtc space
/// @return A reordering of DOFs of a rtc space of the given degree
Eigen::MatrixXd quadrilateral_rtc_reflection(int degree);

/// Rotate the DOFs of a rtc space on a quadrilateral
/// @param degree The number of DOFs along one side of the quadrilateral
/// @return A reordering of DOFs of a rtc space of the given degree
Eigen::MatrixXd quadrilateral_rtc_rotation(int degree);

/// Reflect the DOFs of a nc space on a quadrilateral
/// @param degree The degree of the nc space
/// @return A reordering of DOFs of a nc space of the given degree
Eigen::MatrixXd quadrilateral_nc_reflection(int degree);

/// Rotate the DOFs of a nc space on a quadrilateral
/// @param degree The number of DOFs along one side of the quadrilateral
/// @return A reordering of DOFs of a nc space of the given degree
Eigen::MatrixXd quadrilateral_nc_rotation(int degree);

} // namespace dofperms

} // namespace basix
