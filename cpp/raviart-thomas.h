// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "finite-element.h"

namespace libtab
{
/// Raviart-Thomas element
namespace rt
{
/// @param celltype
/// @param degree
FiniteElement create(cell::Type celltype, int degree);

static std::string family_name = "Raviart-Thomas";
} // namespace rt
} // namespace libtab
