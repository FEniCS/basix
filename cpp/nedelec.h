// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "finite-element.h"

namespace libtab
{
namespace nedelec
{
/// Create Nedelec element (first kind)
/// @param celltype
/// @param degree
FiniteElement create(cell::Type celltype, int degree);

static std::string family_name = "Nedelec 1st kind H(curl)";
} // namespace nedelec
} // namespace libtab
