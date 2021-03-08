// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "finite-element.h"

namespace basix
{
/// Create Raviart-Thomas element
/// @param celltype
/// @param degree
FiniteElement create_rt(cell::type celltype, int degree);

} // namespace basix
