// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "element-families.h"
#include "finite-element.h"

namespace basix::element
{
/// Create a Lagrange element on cell with given degree
/// @param[in] celltype The cell type
/// @param[in] degree The degree of the element
/// @param[in] variant The variant of the element to be created
/// @param[in] discontinuous Controls whether the element is continuous or
/// discontinuous
/// @return A finite element
FiniteElement create_lagrange(cell::type celltype, int degree,
                              element::lagrange_variant variant,
                              bool discontinuous);

/// Create a DPC element on cell with given degree
/// @note DPC elements must be discontinuous
/// @param[in] celltype The cell type
/// @param[in] degree The degree of the element
/// @param[in] discontinuous Controls whether the element is continuous or
/// discontinuous
/// @return A finite element
FiniteElement create_dpc(cell::type celltype, int degree, bool discontinuous);
} // namespace basix::element
