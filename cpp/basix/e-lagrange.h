// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "finite-element.h"

namespace basix
{
/// Create a Lagrange element on cell with given degree
/// @param celltype interval, triangle, quadrilateral, tetrahedral, or
/// hexahedral celltype
/// @param[in] degree
/// @param[in] variant
/// @return A FiniteElement
FiniteElement create_lagrange(cell::type celltype, int degree,
                              element::variant variant);

/// Create a Discontinuous Lagrange element on cell with given degree
/// @param celltype interval, triangle, quadrilateral, tetrahedral, or
/// hexahedral celltype
/// @param[in] degree
/// @param[in] variant
/// @return A FiniteElement
FiniteElement create_dlagrange(cell::type celltype, int degree,
                               element::variant variant);

/// Create a DPC element on cell with given degree
/// @param celltype interval, quadrilateral or hexahedral celltype
/// @param[in] degree
/// @param[in] variant
/// @return A FiniteElement
FiniteElement create_dpc(cell::type celltype, int degree,
                         element::variant variant);
} // namespace basix
