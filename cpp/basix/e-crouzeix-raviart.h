// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "finite-element.h"
#include <concepts>

namespace basix::element
{

/// Crouzeix-Raviart element
/// @note degree must be 1 for Crouzeix-Raviart
/// @param[in] celltype The cell type
/// @param[in] degree The degree of the element
/// @param[in] discontinuous Controls whether the element is continuous or
/// discontinuous
/// @return A finite element
template <std::floating_point T>
FiniteElement<T> create_cr(cell::type celltype, int degree, bool discontinuous);

} // namespace basix::element
