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
/// @param[in] lattice_type The lattice type that should be used to arrange DOF
/// points of the element
/// @return A FiniteElement
FiniteElement create_lagrange(cell::type celltype, int degree,
                              lattice::type lattice_type, bool discontinuous);

/// Create a Discontinuous Lagrange element on cell with given degree
/// @param[in] celltype The reference cell type that the element is defined on
/// @param[in] degree The degree of the element
/// @return A FiniteElement
FiniteElement create_dlagrange(cell::type celltype, int degree,
                               bool discontinuous);

/// Create a DPC element on cell with given degree
/// @param[in] celltype The reference cell type that the element is defined on
/// @param[in] degree The degree of the element
/// @return A FiniteElement
FiniteElement create_dpc(cell::type celltype, int degree, bool discontinuous);
} // namespace basix
