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
  custom,
  P,
  RT,
  N1E,
  BDM,
  N2E,
  CR,
  Regge,
  DP,
  Bubble,
  Serendipity
};

/// Convert string to a family
element::family str_to_type(std::string name);

// Convert family to string
const std::string& type_to_str(element::family type);

} // namespace element

} // namespace basix
