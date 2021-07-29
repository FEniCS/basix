// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "finite-element.h"

namespace basix
{
/// Create a GLL Lagrange element on cell with given degree
/// @param celltype interval, triangle, quadrilateral, tetrahedral, or
/// hexahedral celltype
/// @param[in] degree
/// @return A FiniteElement
FiniteElement create_gll(cell::type celltype, int degree);

} // namespace basix
