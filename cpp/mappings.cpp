// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "mappings.h"
#include <map>
#include <stdexcept>

using namespace basix;

//-----------------------------------------------------------------------------
Eigen::ArrayXXd mapping::apply_mapping(Eigen::ArrayXXd data, mapping::type mapping_type, int value_size)
{
  switch (mapping_type)
  {
  case mapping::type::identity:
    return data;
  default:
    throw std::runtime_error("Mapping not yet implemented");
  }
}
//-----------------------------------------------------------------------------
const std::string& mapping::type_to_str(mapping::type type)
{
  static const std::map<mapping::type, std::string> type_to_name
      = {{mapping::type::identity, "identity"},
         {mapping::type::covariantPiola, "covariant Piola"},
         {mapping::type::contravariantPiola, "contravariant Piola"},
         {mapping::type::doubleCovariantPiola, "double covariant Piola"},
         {mapping::type::doubleContravariantPiola, "double contravariant Piola"}};

  auto it = type_to_name.find(type);
  if (it == type_to_name.end())
    throw std::runtime_error("Can't find type");

  return it->second;
}
