// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "element-families.h"
#include "finite-element.h"
#include "lattice.h"

namespace basix
{
/// Create a Lagrange element on cell with given degree
/// @param[in] celltype The reference cell type that the element is defined on
/// @param[in] degree The degree of the element
/// @param[in] variant The variant of the element to be created
/// @param[in] simplex_method The simplex method that should be used to create
/// lattice on simplices
/// @param discontinuous
/// points of the element
/// @return A FiniteElement
FiniteElement create_lagrange(cell::type celltype, int degree,
                              element::lagrange_variant variant,
                              bool discontinuous);

/// Create a DPC element on cell with given degree
/// @param[in] celltype The reference cell type that the element is defined on
/// @param[in] degree The degree of the element
/// @param discontinuous
/// @return A FiniteElement
FiniteElement create_dpc(cell::type celltype, int degree, bool discontinuous);
} // namespace basix
