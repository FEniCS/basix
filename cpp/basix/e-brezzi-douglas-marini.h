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
/// @param discontinuous
FiniteElement create_bdm(cell::type celltype, int degree, bool discontinuous);

} // namespace basix
