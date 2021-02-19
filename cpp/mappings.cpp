// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "mappings.h"
#include <functional>
#include <map>
#include <stdexcept>

namespace
{
Eigen::ArrayXd identity(const Eigen::ArrayXd& reference_data,
                        const Eigen::MatrixXd& J, const double detJ,
                        const Eigen::MatrixXd& K)
{
  return reference_data;
}
Eigen::ArrayXd covariant_piola(const Eigen::ArrayXd& reference_data,
                               const Eigen::MatrixXd& J, const double detJ,
                               const Eigen::MatrixXd& K)
{
  return K.transpose() * reference_data.matrix();
}
Eigen::ArrayXd contravariant_piola(const Eigen::ArrayXd& reference_data,
                                   const Eigen::MatrixXd& J, const double detJ,
                                   const Eigen::MatrixXd& K)
{
  return 1 / detJ * J * reference_data.matrix();
}
Eigen::ArrayXd double_covariant_piola(const Eigen::ArrayXd& reference_data,
                                      const Eigen::MatrixXd& J,
                                      const double detJ,
                                      const Eigen::MatrixXd& K)
{
  Eigen::Map<const Eigen::MatrixXd> data_matrix(reference_data.data(), J.cols(),
                                                J.cols());
  Eigen::MatrixXd result = K.transpose() * data_matrix * K;
  return Eigen::Map<Eigen::ArrayXd>(result.data(), J.rows() * J.rows());
}
Eigen::ArrayXd double_contravariant_piola(const Eigen::ArrayXd& reference_data,
                                          const Eigen::MatrixXd& J,
                                          const double detJ,
                                          const Eigen::MatrixXd& K)
{
  Eigen::Map<const Eigen::MatrixXd> data_matrix(reference_data.data(), J.cols(),
                                                J.cols());
  Eigen::MatrixXd result = 1 / (detJ * detJ) * J * data_matrix * J.transpose();
  return Eigen::Map<Eigen::ArrayXd>(result.data(), J.rows() * J.rows());
}
} // namespace

using namespace basix;

//-----------------------------------------------------------------------------
std::function<Eigen::ArrayXd(const Eigen::ArrayXd&, const Eigen::MatrixXd&,
                             const double, const Eigen::MatrixXd&)>
mapping::get_forward_map(mapping::type mapping_type)
{
  switch (mapping_type)
  {
  case mapping::type::identity:
    return identity;
  case mapping::type::covariantPiola:
    return covariant_piola;
  case mapping::type::contravariantPiola:
    return contravariant_piola;
  case mapping::type::doubleCovariantPiola:
    return double_covariant_piola;
  case mapping::type::doubleContravariantPiola:
    return double_contravariant_piola;
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
