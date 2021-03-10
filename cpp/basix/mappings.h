// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "span.hpp"
#include <string>
#include <vector>
#include <xtensor/xtensor.hpp>

/// Information about mappings.
namespace basix::mapping
{

/// Cell type
enum class type
{
  identity,
  covariantPiola,
  contravariantPiola,
  doubleCovariantPiola,
  doubleContravariantPiola,
};

/// Get the function that maps data from the reference to the physical
/// cell
/// @param mapping_type Mapping type
/// @return The mapping function
std::function<std::vector<double>(const tcb::span<const double>&,
                                  const xt::xtensor<double, 2>&, double,
                                  const xt::xtensor<double, 2>&)>
get_forward_map(mapping::type mapping_type);

/// Convert mapping type enum to string
const std::string& type_to_str(mapping::type type);

} // namespace basix::mapping
