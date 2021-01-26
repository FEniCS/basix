// Copyright (c) 2020 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once
#include <string>

namespace basix
{

namespace element
{
/// Enum of available element families
enum class family
{
  P,
  RT,
  N1E,
  BDM,
  N2E,
  CR,
  Regge,
  DP
};

/// Convert string to a family
element::family str_to_family(std::string name);

// Convert family to string
const std::string& family_to_str(element::family type);

} // namespace element

} // namespace basix
