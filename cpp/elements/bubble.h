// Copyright (c) 2020 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "core/cell.h"
#include "core/finite-element.h"
#include <string>

namespace basix
{
/// Create a bubble element on cell with given degree
/// @param[in] celltype interval, triangle, tetrahedral, quadrilateral or
/// hexahedral celltype
/// @param[in] degree
/// @return A FiniteElement
FiniteElement create_bubble(cell::type celltype, int degree);
} // namespace basix
