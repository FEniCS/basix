// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include <string>
#include <Eigen/Dense>

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

/// Apply mapping
/// @param data The data to apply the mapping to
/// @param mapping_type Mapping type
/// @param value_size The value size of the data
/// @return The mapped data
// TODO: should data be in/out?
Eigen::ArrayXXd apply_mapping(Eigen::ArrayXXd data, mapping::type mapping_type, int value_size);

/// Convert mapping type enum to string
const std::string& type_to_str(mapping::type type);

} // namespace mapping
} // namespace basix
