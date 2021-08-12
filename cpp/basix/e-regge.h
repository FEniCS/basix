// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "finite-element.h"

namespace basix
{
/// Create Regge element
/// @param celltype
/// @param degree
/// @param discontinuous
FiniteElement create_regge(cell::type celltype, int degree, bool discontinuous);

} // namespace basix
