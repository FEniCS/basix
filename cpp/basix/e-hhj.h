// Copyright (c) 2022 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "finite-element.h"
#include <concepts>

namespace basix::element
{
/// Create Hellan-Herrmann-Johnson element
/// @param[in] celltype The cell type
/// @param[in] degree The degree of the element
/// @param[in] discontinuous Controls whether the element is continuous or
/// discontinuous
/// @return A finite element
template <std::floating_point T>
FiniteElement<T> create_hhj(cell::type celltype, int degree,
                            bool discontinuous);

} // namespace basix::element
