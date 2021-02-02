// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "finite-element.h"
#include <string>

namespace basix
{
/// Create BDM element
/// @param celltype
/// @param degree
FiniteElement create_bdm(cell::type celltype, int degree);

} // namespace basix
