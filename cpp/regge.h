// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "finite-element.h"

namespace libtab
{
/// Regge element
namespace regge
{
/// Create Regge element
/// @param celltype
/// @param degree
FiniteElement create(cell::Type celltype, int degree);

static std::string family_name = "Regge";
} // namespace regge
} // namespace libtab
