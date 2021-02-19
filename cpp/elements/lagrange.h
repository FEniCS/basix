// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "core/cell.h"
#include "core/finite-element.h"
#include <string>

namespace basix
{
/// Create a Lagrange element on cell with given degree
/// @param[in] celltype interval, triangle or tetrahedral celltype
/// @param[in] degree
/// @return A FiniteElement
FiniteElement create_lagrange(cell::type celltype, int degree);

/// Create a Discontinuous Lagrange element on cell with given degree
/// @param celltype interval, triangle or tetrahedral celltype
/// @param[in] degree
/// @return A FiniteElement
FiniteElement create_dlagrange(cell::type celltype, int degree);
} // namespace basix
