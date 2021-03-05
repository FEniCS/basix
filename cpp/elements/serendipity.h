// Copyright (c) 2020 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "core/cell.h"
#include "core/finite-element.h"
#include <string>

namespace basix
{
/// Create a serendipity element on cell with given degree
/// @param[in] celltype quadrilateral or hexahedral celltype
/// @param[in] degree
/// @return A FiniteElement
FiniteElement create_serendipity(cell::type celltype, int degree);
} // namespace basix
