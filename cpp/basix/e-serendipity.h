// Copyright (c) 2020 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "element-families.h"
#include "finite-element.h"

namespace basix::element
{
/// Create a serendipity element on cell with given degree
/// @param[in] celltype The cell type
/// @param[in] degree The degree of the element
/// @param[in] discontinuous Controls whether the element is continuous or
/// discontinuous
/// @return A finite element
FiniteElement create_serendipity(cell::type celltype, int degree,
                                 bool discontinuous);

/// Create a serendipity element on cell with given degree
/// @param[in] celltype The cell type
/// @param[in] degree The degree of the element
/// @param[in] variant The variant of the element to be created
/// @param[in] discontinuous Controls whether the element is continuous or
/// discontinuous
/// @return A finite element
FiniteElement create_serendipity(cell::type celltype, int degree,
                                 element::lagrange_variant variant,
                                 bool discontinuous);

/// Create a serendipity H(div) element on cell with given degree
/// @param[in] celltype The cell type
/// @param[in] degree The degree of the element
/// @param[in] discontinuous Controls whether the element is continuous or
/// discontinuous
/// @return A finite element
FiniteElement create_serendipity_div(cell::type celltype, int degree,
                                     bool discontinuous);

/// Create a serendipity H(curl) element on cell with given degree
/// @param[in] celltype The cell type
/// @param[in] degree The degree of the element
/// @param[in] discontinuous Controls whether the element is continuous or
/// discontinuous
/// @return A finite element
FiniteElement create_serendipity_curl(cell::type celltype, int degree,
                                      bool discontinuous);
} // namespace basix::element
