// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "finite-element.h"
#include "lattice.h"

namespace basix
{
/// An enum defining the variants of a Lagrange space that can be created
enum class lagrange_variant
{
  equispaced = 0,
  gll_warped = 1,
  gll_isaac = 2,
  chebyshev_warped = 3,
  chebyshev_isaac = 4,
};

/// Create a Lagrange element on cell with given degree
/// @param[in] celltype The reference cell type that the element is defined on
/// @param[in] degree The degree of the element
/// @param[in] lattice_type The lattice type that should be used to arrange DOFs
/// @param[in] simplex_method The simplex method that should be used to create
/// lattice on simplices
/// @param discontinuous
/// points of the element
/// @return A FiniteElement
FiniteElement create_lagrange(cell::type celltype, int degree,
                              lagrange_variant variant, bool discontinuous);

/// Create a DPC element on cell with given degree
/// @param[in] celltype The reference cell type that the element is defined on
/// @param[in] degree The degree of the element
/// @param discontinuous
/// @return A FiniteElement
FiniteElement create_dpc(cell::type celltype, int degree, bool discontinuous);
} // namespace basix
