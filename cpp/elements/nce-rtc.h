// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "core/cell.h"
#include "core/finite-element.h"

namespace basix
{
/// Create RTC H(div) element
/// @param celltype
/// @param degree
FiniteElement create_rtc(cell::type celltype, int degree);

/// Create NC H(curl) element
/// @param celltype
/// @param degree
FiniteElement create_nce(cell::type celltype, int degree);

namespace dofperms
{
/// Reflect the DOFs of a RTC H(div) space on a quadrilateral
/// @param degree The degree of the RTC H(div) space
/// @return A reordering of DOFs of a RTC H(div) space of the given degree
Eigen::MatrixXd quadrilateral_rtc_reflection(int degree);

/// Rotate the DOFs of a RTC H(div) space on a quadrilateral
/// @param degree The number of DOFs along one side of the quadrilateral
/// @return A reordering of DOFs of a RTC H(div) space of the given degree
Eigen::MatrixXd quadrilateral_rtc_rotation(int degree);

/// Reflect the DOFs of a NC H(curl) space on a quadrilateral
/// @param degree The degree of the NC H(curl) space
/// @return A reordering of DOFs of a NC H(curl) space of the given degree
Eigen::MatrixXd quadrilateral_nce_reflection(int degree);

/// Rotate the DOFs of a NC H(curl) space on a quadrilateral
/// @param degree The number of DOFs along one side of the quadrilateral
/// @return A reordering of DOFs of a NC H(curl) space of the given degree
Eigen::MatrixXd quadrilateral_nce_rotation(int degree);

} // namespace dofperms

} // namespace basix
