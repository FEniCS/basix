// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "libtab.h"

namespace libtab
{

/// Crouzeix-Raviart element
namespace cr
{
/// @note degree must be 1 for Crouzeix-Raviart
/// @param celltype
/// @param degree
FiniteElement create(cell::type celltype, int degree);

static std::string family_name = "Crouzeix-Raviart";
} // namespace cr
} // namespace libtab
