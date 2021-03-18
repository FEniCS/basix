// Copyright (c) 2021 Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "maps.h"
#include <map>
#include <stdexcept>

//-----------------------------------------------------------------------------
const std::string& basix::maps::type_to_str(maps::type type)
{
  static const std::map<maps::type, std::string> type_to_name
      = {{maps::type::identity, "identity"},
         {maps::type::covariantPiola, "covariant Piola"},
         {maps::type::contravariantPiola, "contravariant Piola"},
         {maps::type::doubleCovariantPiola, "double covariant Piola"},
         {maps::type::doubleContravariantPiola, "double contravariant Piola"}};

  auto it = type_to_name.find(type);
  if (it == type_to_name.end())
    throw std::runtime_error("Can't find type");

  return it->second;
}
//-----------------------------------------------------------------------------
