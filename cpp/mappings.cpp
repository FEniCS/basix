// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "mappings.h"
#include <map>
#include <stdexcept>

using namespace basix;

//-----------------------------------------------------------------------------
Eigen::ArrayXd mapping::map_push_forward(const Eigen::ArrayXd& reference_data,
                                         const Eigen::MatrixXd& J, double detJ,
                                         const Eigen::MatrixXd& K,
                                         mapping::type mapping_type,
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
    Eigen::Map<const Eigen::MatrixXd> data_matrix(
        reference_data.data(), value_shape[0], value_shape[1]);
    Eigen::MatrixXd result = K.transpose() * data_matrix * K;
    return Eigen::Map<Eigen::ArrayXd>(result.data(), reference_data.size());
  }
  case mapping::type::doubleContravariantPiola:
  {
    Eigen::Map<const Eigen::MatrixXd> data_matrix(
        reference_data.data(), value_shape[0], value_shape[1]);
    Eigen::MatrixXd result
        = 1 / (detJ * detJ) * J * data_matrix * J.transpose();
    return Eigen::Map<Eigen::ArrayXd>(result.data(), reference_data.size());
  }
  default:
    throw std::runtime_error("Mapping not yet implemented");
  }
}
//-----------------------------------------------------------------------------
Eigen::ArrayXd mapping::map_pull_back(const Eigen::ArrayXd& physical_data,
                                      const Eigen::MatrixXd& J, double detJ,
                                      const Eigen::MatrixXd& K,
                                      mapping::type mapping_type,
                                      const std::vector<int> value_shape)
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
    Eigen::Map<const Eigen::MatrixXd> data_matrix(
        physical_data.data(), value_shape[0], value_shape[1]);
    Eigen::MatrixXd result = J.transpose() * data_matrix * J;
    return Eigen::Map<Eigen::ArrayXd>(result.data(), physical_data.size());
  }
  case mapping::type::doubleContravariantPiola:
  {
    Eigen::Map<const Eigen::MatrixXd> data_matrix(
        physical_data.data(), value_shape[0], value_shape[1]);
    Eigen::MatrixXd result = detJ * detJ * K * data_matrix * K.transpose();
    return Eigen::Map<Eigen::ArrayXd>(result.data(), physical_data.size());
  }
  default:
    throw std::runtime_error("Mapping not yet implemented");
  }
}
//-----------------------------------------------------------------------------
const std::string& mapping::type_to_str(mapping::type type)
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
