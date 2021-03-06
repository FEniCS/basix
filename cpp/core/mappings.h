// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "span.hpp"
#include <Eigen/Dense>
#include <string>
#include <vector>

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

/// Get the function that maps data from the reference to the physical cell.
/// @param mapping_type Mapping type
/// @return The mapping function
std::function<std::vector<double>(const tcb::span<const double>&,
                                  const Eigen::MatrixXd&, double,
                                  const Eigen::MatrixXd&)>
get_forward_map(mapping::type mapping_type);

/// Convert mapping type enum to string
const std::string& type_to_str(mapping::type type);

} // namespace basix::mapping
