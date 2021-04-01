// Copyright (c) 2020 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "finite-element.h"

namespace basix
{
/// Create a serendipity element on cell with given degree
/// @param[in] celltype quadrilateral or hexahedral celltype
/// @param[in] degree
/// @return A FiniteElement
FiniteElement create_serendipity(cell::type celltype, int degree);

/// Create a serendipity H(div) element on cell with given degree
/// @param[in] celltype quadrilateral or hexahedral celltype
/// @param[in] degree
/// @return A FiniteElement
FiniteElement create_serendipity_div(cell::type celltype, int degree);

/// Create a serendipity H(curl) element on cell with given degree
/// @param[in] celltype quadrilateral or hexahedral celltype
/// @param[in] degree
/// @return A FiniteElement
FiniteElement create_serendipity_curl(cell::type celltype, int degree);
} // namespace basix
