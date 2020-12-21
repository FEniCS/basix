// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "finite-element.h"
#include <string>

namespace basix
{
/// Create a Lagrange element on cell with given degree
/// @param[in] celltype interval, triangle or tetrahedral celltype
/// @param[in] degree
/// @param[in] name Identifier string (optional)
/// @return A FiniteElemenet
FiniteElement create_lagrange(cell::type celltype, int degree,
                              const std::string& name = std::string());

/// Create a Discontinuous Lagrange element on cell with given degree
/// @param celltype interval, triangle or tetrahedral celltype
/// @param[in] degree
/// @param[in] name Identifier string (optional)
/// @return A FiniteElemenet
FiniteElement create_dlagrange(cell::type celltype, int degree,
                               const std::string& name = std::string());
} // namespace basix
