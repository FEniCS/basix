// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "finite-element.h"
#include <string>

namespace basix
{
/// Create Nedelec element (first kind)
/// @param celltype
/// @param degree
/// @param name
FiniteElement create_nedelec(cell::type celltype, int degree,
                             const std::string& name = std::string());

// static std::string family_name = "Nedelec 1st kind H(curl)";
// } // namespace basix

/// Create Nedelec element (second kind)
/// @param celltype
/// @param degree
/// @param name
FiniteElement create_nedelec2(cell::type celltype, int degree,
                              const std::string& name = std::string());

// static std::string family_name = "Nedelec 2nd kind H(curl)";
// } // namespace nedelec2

} // namespace basix
