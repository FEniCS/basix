// Copyright (c) 2020 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once
#include "lattice.h"
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
  DPC,
  Bubble,
  Serendipity
};

/// Enum of available element variants
enum class variant
{
  DEFAULT,
  EQ,
  GLL
};

/// Convert string to a family
element::family str_to_family(std::string name);

// Convert family to string
const std::string& family_to_str(element::family family);

/// Convert string to a variant
element::variant str_to_variant(std::string name);

// Convert variant to string
const std::string& variant_to_str(element::variant variant);

// Convert variant to lattice type
lattice::type variant_to_lattice(element::variant variant);

// Convert variant to lattice type
element::variant lattice_to_variant(lattice::type lattice_type);

} // namespace element

} // namespace basix
