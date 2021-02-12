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

/// Apply mapping
/// @param reference_data The data to apply the mapping to
/// @param J The Jacobian
/// @param detJ The determinant of the Jacobian
/// @param K The inverse of the Jacobian
/// @param mapping_type Mapping type
/// @return The mapped data
// TODO: should data be in/out?
Eigen::ArrayXd map_push_forward(const Eigen::ArrayXd& reference_data,
                                const Eigen::MatrixXd& J, double detJ,
                                const Eigen::MatrixXd& K,
                                mapping::type mapping_type);

/// Convert mapping type enum to string
const std::string& type_to_str(mapping::type type);

} // namespace mapping
} // namespace basix
