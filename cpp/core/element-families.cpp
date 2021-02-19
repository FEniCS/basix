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
         {"Lagrange", element::family::P},
         {"Discontinuous Lagrange", element::family::DP},
         {"Brezzi-Douglas-Marini", element::family::BDM},
         {"Raviart-Thomas", element::family::RT},
         {"Nedelec 1st kind H(curl)", element::family::N1E},
         {"Nedelec 2nd kind H(curl)", element::family::N2E},
         {"Regge", element::family::Regge},
         {"Crouzeix-Raviart", element::family::CR}};

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
         {element::family::P, "Lagrange"},
         {element::family::DP, "Discontinuous Lagrange"},
         {element::family::BDM, "Brezzi-Douglas-Marini"},
         {element::family::RT, "Raviart-Thomas"},
         {element::family::N1E, "Nedelec 1st kind H(curl)"},
         {element::family::N2E, "Nedelec 2nd kind H(curl)"},
         {element::family::Regge, "Regge"},
         {element::family::CR, "Crouzeix-Raviart"}};

  auto it = name_to_type.find(type);
  if (it == name_to_type.end())
    throw std::runtime_error("Can't find type");

  return it->second;
}
//-----------------------------------------------------------------------------
