// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "finite-element.h"
#include <string>

namespace basix
{
/// Create BDM element
/// @param celltype
/// @param degree
/// @param name
FiniteElement create_bdm(cell::type celltype, int degree,
                         const std::string& name = std::string());

} // namespace basix
