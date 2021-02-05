// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include <Eigen/Dense>
#include <map>
#include <stdexcept>
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

/// Apply mapping
/// @param reference_data The data to apply the mapping to
/// @param J The Jacobian
/// @param detJ The determinant of the Jacobian
/// @param K The inverse of the Jacobian
/// @param mapping_type Mapping type
/// @param value_shape The value shape of the data
/// @return The mapped data
// TODO: should data be in/out?
template <typename T>
Eigen::Array<T, Eigen::Dynamic, 1>
map_push_forward(const Eigen::Array<T, Eigen::Dynamic, 1>& reference_data,
                 const Eigen::MatrixXd& J, double detJ,
                 const Eigen::MatrixXd& K, mapping::type mapping_type,
                 const std::vector<int> value_shape)
{
  switch (mapping_type)
  {
  case mapping::type::identity:
    return reference_data;
  case mapping::type::covariantPiola:
    return K.transpose() * reference_data.matrix();
  case mapping::type::contravariantPiola:
    return 1 / detJ * J * reference_data.matrix();
  case mapping::type::doubleCovariantPiola:
  {
    Eigen::Map<
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        data_matrix(reference_data.data(), value_shape[0], value_shape[1]);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result
        = K.transpose() * data_matrix * K;
    return Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>(
        result.data(), reference_data.size());
  }
  case mapping::type::doubleContravariantPiola:
  {
    Eigen::Map<
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        data_matrix(reference_data.data(), value_shape[0], value_shape[1]);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result
        = 1 / (detJ * detJ) * J * data_matrix * J.transpose();
    return Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>(
        result.data(), reference_data.size());
  }
  default:
    throw std::runtime_error("Mapping not yet implemented");
  }
}

/// Apply inverse mapping
/// @param physical_data The data to apply the inverse mapping to
/// @param J The Jacobian
/// @param detJ The determinant of the Jacobian
/// @param K The inverse of the Jacobian
/// @param mapping_type Mapping type
/// @param value_shape The value shape of the data
/// @return The mapped data
// TODO: should data be in/out?
template <typename T>
Eigen::Array<T, Eigen::Dynamic, 1>
map_pull_back(const Eigen::Array<T, Eigen::Dynamic, 1>& physical_data,
              const Eigen::MatrixXd& J, double detJ, const Eigen::MatrixXd& K,
              mapping::type mapping_type, const std::vector<int> value_shape)
{
  switch (mapping_type)
  {
  case mapping::type::identity:
    return physical_data;
  case mapping::type::covariantPiola:
    return J.transpose() * physical_data.matrix();
  case mapping::type::contravariantPiola:
    return detJ * K * physical_data.matrix();
  case mapping::type::doubleCovariantPiola:
  {
    Eigen::Map<
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        data_matrix(physical_data.data(), value_shape[0], value_shape[1]);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result
        = J.transpose() * data_matrix * J;
    return Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>(result.data(),
                                                          physical_data.size());
  }
  case mapping::type::doubleContravariantPiola:
  {
    Eigen::Map<
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        data_matrix(physical_data.data(), value_shape[0], value_shape[1]);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> result
        = detJ * detJ * K * data_matrix * K.transpose();
    return Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>(result.data(),
                                                          physical_data.size());
  }
  default:
    throw std::runtime_error("Mapping not yet implemented");
  }
}

/// Convert mapping type enum to string
inline const std::string& type_to_str(mapping::type type)
{
  static const std::map<mapping::type, std::string> type_to_name = {
      {mapping::type::identity, "identity"},
      {mapping::type::covariantPiola, "covariant Piola"},
      {mapping::type::contravariantPiola, "contravariant Piola"},
      {mapping::type::doubleCovariantPiola, "double covariant Piola"},
      {mapping::type::doubleContravariantPiola, "double contravariant Piola"}};

  auto it = type_to_name.find(type);
  if (it == type_to_name.end())
    throw std::runtime_error("Can't find type");

  return it->second;
}

} // namespace basix::mapping
