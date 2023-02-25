// Copyright (c) 2020-2022 Chris Richardson, Matthew Scroggs and Garth N. Wells
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "element-families.h"
#include "finite-element.h"

namespace basix::element
{
/// @brief Create a Lagrange(-like) element on cell with given degree
/// @param[in] celltype The element cell type
/// @param[in] degree The degree of the element
/// @param[in] variant The variant of the element to be created
/// @param[in] discontinuous True if the is discontinuous
/// @return A finite element
FiniteElement create_lagrange(cell::type celltype, int degree,
                              lagrange_variant variant, bool discontinuous);
} // namespace basix::element
