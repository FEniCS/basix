// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "finite-element.h"

namespace basix
{

/// Crouzeix-Raviart element
/// @note degree must be 1 for Crouzeix-Raviart
/// @param celltype
/// @param degree
FiniteElement create_cr(cell::type celltype, int degree);

} // namespace basix
