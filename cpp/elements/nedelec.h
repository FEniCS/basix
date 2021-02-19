// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "core/finite-element.h"
#include <string>

namespace basix
{
/// Create Nedelec element (first kind)
/// @param celltype
/// @param degree
FiniteElement create_nedelec(cell::type celltype, int degree);

/// Create Nedelec element (second kind)
/// @param celltype
/// @param degree
FiniteElement create_nedelec2(cell::type celltype, int degree);

} // namespace basix
