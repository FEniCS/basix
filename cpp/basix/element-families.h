// Copyright (c) 2020 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once
#include <string>

namespace basix
{

namespace element
{
/// An enum defining the variants of a Lagrange space that can be created
enum class lagrange_variant
{
  equispaced = 0,
  gll_warped = 1,
  gll_isaac = 2,
  chebyshev_warped = 3,
  chebyshev_isaac = 4,
};

/// Enum of available element families
enum class family
{
  custom = 0,
  P = 1,
  RT = 2,
  N1E = 3,
  BDM = 4,
  N2E = 5,
  CR = 6,
  Regge = 7,
  DPC = 8,
  Bubble = 9,
  Serendipity = 10
};

/// Convert string to a family
element::family str_to_type(std::string name);

// Convert family to string
const std::string& type_to_str(element::family type);

} // namespace element

} // namespace basix
