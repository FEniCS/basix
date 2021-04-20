// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "finite-element.h"

namespace basix
{
/// Create Nedelec element (first kind)
/// @param celltype
/// @param degree
/// @param variant
FiniteElement create_nedelec(cell::type celltype, int degree,
                             element::variant variant);

/// Create Nedelec element (second kind)
/// @param celltype
/// @param degree
/// @param variant
FiniteElement create_nedelec2(cell::type celltype, int degree,
                              element::variant variant);

} // namespace basix
