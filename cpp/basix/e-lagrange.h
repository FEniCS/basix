// Copyright (c) 2020-2022 Chris Richardson, Matthew Scroggs and Garth N. Wells
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "element-families.h"
#include "finite-element.h"
#include <concepts>

namespace basix::element
{
/// @brief Create a Lagrange(-like) element on cell with given degree
/// @param[in] celltype The element cell type
/// @param[in] degree The degree of the element
/// @param[in] variant The variant of the element to be created
/// @param[in] discontinuous True if the is discontinuous
/// @param[in] dof_ordering DOF reordering
/// @return A finite element
template <std::floating_point T>
FiniteElement<T> create_lagrange(cell::type celltype, int degree,
                                 lagrange_variant variant, bool discontinuous,
                                 std::vector<int> dof_ordering = {});

/// @brief Create an iso macro element on cell with given degree
/// @param[in] celltype The element cell type
/// @param[in] degree The degree of the element
/// @param[in] variant The variant of the element to be created
/// @param[in] discontinuous True if the is discontinuous
/// @return A finite element
template <std::floating_point T>
FiniteElement<T> create_iso(cell::type celltype, int degree,
                            lagrange_variant variant, bool discontinuous);
} // namespace basix::element
