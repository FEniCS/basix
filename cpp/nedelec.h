// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "libtab.h"
#include <string>

namespace libtab
{
/// Create Nedelec element (first kind)
/// @param celltype
/// @param degree
FiniteElement create_nedelec(cell::type celltype, int degree,
                             const std::string& name = std::string());

// static std::string family_name = "Nedelec 1st kind H(curl)";
// } // namespace libtab

/// Create Nedelec element (second kind)
/// @param celltype
/// @param degree
FiniteElement create_nedelec2(cell::type celltype, int degree,
                              const std::string& name = std::string());

// static std::string family_name = "Nedelec 2nd kind H(curl)";
// } // namespace nedelec2

} // namespace libtab
