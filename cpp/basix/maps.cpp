// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "maps.h"
#include <map>
#include <stdexcept>

//-----------------------------------------------------------------------------
const std::string& basix::map::type_to_str(map::type type)
{
  static const std::map<map::type, std::string> type_to_name
      = {{map::type::identity, "identity"},
         {map::type::covariantPiola, "covariant Piola"},
         {map::type::contravariantPiola, "contravariant Piola"},
         {map::type::doubleCovariantPiola, "double covariant Piola"},
         {map::type::doubleContravariantPiola, "double contravariant Piola"}};

  auto it = type_to_name.find(type);
  if (it == type_to_name.end())
    throw std::runtime_error("Can't find type");

  return it->second;
}
//-----------------------------------------------------------------------------
