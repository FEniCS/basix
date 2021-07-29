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
/// @param[in] celltype interval, triangle, quadrilateral, tetrahedral, or
/// hexahedral celltype
/// @param[in] degree
/// @param[in] lattice_type
/// @return A FiniteElement
FiniteElement create_lagrange(cell::type celltype, int degree,
                              lattice::type lattice_type
                              = lattice::type::gll_warped);

/// Create a Discontinuous Lagrange element on cell with given degree
/// @param[in] celltype interval, triangle, quadrilateral, tetrahedral, or
/// hexahedral celltype
/// @param[in] degree
/// @return A FiniteElement
FiniteElement create_dlagrange(cell::type celltype, int degree);

/// Create a DPC element on cell with given degree
/// @param[in] celltype interval, quadrilateral or hexahedral celltype
/// @param[in] degree
/// @return A FiniteElement
FiniteElement create_dpc(cell::type celltype, int degree);
} // namespace basix
