// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "element-families.h"
#include <map>
#include <stdexcept>

using namespace basix;

//-----------------------------------------------------------------------------
element::family element::str_to_type(std::string name)
{
  static const std::map<std::string, element::family> name_to_type
      = {{"Custom element", element::family::custom},
         {"P", element::family::P},
         {"DP", element::family::DP},
         {"DPC", element::family::DPC},
         {"BDM", element::family::BDM},
         {"RT", element::family::RT},
         {"N1E", element::family::N1E},
         {"N2E", element::family::N2E},
         {"Regge", element::family::Regge},
         {"CR", element::family::CR},
         {"Bubble", element::family::Bubble},
         {"Serendipity", element::family::Serendipity}};

  auto it = name_to_type.find(name);
  if (it == name_to_type.end())
    throw std::runtime_error("Can't find name " + name);

  return it->second;
}
//-----------------------------------------------------------------------------
const std::string& element::type_to_str(element::family type)
{
  static const std::map<element::family, std::string> name_to_type
      = {{element::family::custom, "Custom element"},
         {element::family::P, "P"},
         {element::family::DP, "DP"},
         {element::family::DPC, "DPC"},
         {element::family::BDM, "BDM"},
         {element::family::RT, "RT"},
         {element::family::N1E, "N1E"},
         {element::family::N2E, "N2E"},
         {element::family::Regge, "Regge"},
         {element::family::CR, "CR"},
         {element::family::Bubble, "Bubble"},
         {element::family::Serendipity, "Serendipity"}};

  auto it = name_to_type.find(type);
  if (it == name_to_type.end())
    throw std::runtime_error("Can't find type");

  return it->second;
}
//-----------------------------------------------------------------------------
