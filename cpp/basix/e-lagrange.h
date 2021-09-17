// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "finite-element.h"
#include "lattice.h"

namespace basix
{
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
                              lattice::type lattice_type,
                              lattice::simplex_method simplex_method,
                              bool discontinuous);

/// Create a DPC element on cell with given degree
/// @param[in] celltype The reference cell type that the element is defined on
/// @param[in] degree The degree of the element
/// @param discontinuous
/// @return A FiniteElement
FiniteElement create_dpc(cell::type celltype, int degree, bool discontinuous);
} // namespace basix
