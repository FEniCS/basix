// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "finite-element.h"

namespace basix
{
/// Create BDM element
/// @param celltype
/// @param degree
/// @param variant
FiniteElement create_bdm(cell::type celltype, int degree,
                         element::variant variant);

} // namespace basix
