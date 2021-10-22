// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "finite-element.h"

namespace basix::element
{
/// Create Nedelec element (first kind)
/// @param[in] celltype The cell type
/// @param[in] degree The degree of the element
/// @param[in] discontinuous Controls whether the element is continuous or
/// discontinuous
/// @return A finite element
FiniteElement create_nedelec(cell::type celltype, int degree,
                             bool discontinuous);

/// Create Nedelec element (second kind)
/// @param[in] celltype The cell type
/// @param[in] degree The degree of the element
/// @param[in] discontinuous Controls whether the element is continuous or
/// discontinuous
/// @return A finite element
FiniteElement create_nedelec2(cell::type celltype, int degree,
                              bool discontinuous);

} // namespace basix::element
