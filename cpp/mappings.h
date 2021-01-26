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
Eigen::ArrayXd apply_mapping(int order, const Eigen::ArrayXd& reference_data,
                              const Eigen::MatrixXd& J, double detJ,
                              const Eigen::MatrixXd& K,
                              mapping::type mapping_type,
                                       const std::vector<int> value_shape={1});

/// Convert mapping type enum to string
const std::string& type_to_str(mapping::type type);

} // namespace mapping
} // namespace basix
