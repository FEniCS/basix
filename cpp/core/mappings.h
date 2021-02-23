// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace basix
{

/// Information about mappings.

namespace mapping
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
std::function<Eigen::ArrayXd(const Eigen::ArrayXd&, const Eigen::MatrixXd&,
                             const double, const Eigen::MatrixXd&)>
get_forward_map(mapping::type mapping_type);

/// Convert mapping type enum to string
const std::string& type_to_str(mapping::type type);

} // namespace mapping
} // namespace basix
