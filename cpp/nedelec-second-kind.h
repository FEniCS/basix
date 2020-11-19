// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "finite-element.h"

namespace libtab
{
/// Nedelec element (second kind)
namespace nedelec2
{
/// Create Nedelec element (second kind)
/// @param celltype
/// @param degree
FiniteElement create(cell::Type celltype, int degree);

static std::string family_name = "Nedelec 2nd kind H(curl)";
} // namespace nedelec2
} // namespace libtab
