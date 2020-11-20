// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "libtab.h"

namespace libtab
{
/// Create Regge element
/// @param celltype
/// @param degree
FiniteElement create_regge(cell::type celltype, int degree,
                           const std::string& name = std::string());

} // namespace libtab
