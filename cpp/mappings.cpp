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
                                         mapping::type mapping_type)
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
    Eigen::Map<const Eigen::MatrixXd> data_matrix(reference_data.data(),
                                                  J.cols(), J.cols());
    Eigen::MatrixXd result = K.transpose() * data_matrix * K;
    return Eigen::Map<Eigen::ArrayXd>(result.data(), J.rows() * J.rows());
  }
  case mapping::type::doubleContravariantPiola:
  {
    Eigen::Map<const Eigen::MatrixXd> data_matrix(reference_data.data(),
                                                  J.cols(), J.cols());
    Eigen::MatrixXd result
        = 1 / (detJ * detJ) * J * data_matrix * J.transpose();
    return Eigen::Map<Eigen::ArrayXd>(result.data(), J.rows() * J.rows());
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
