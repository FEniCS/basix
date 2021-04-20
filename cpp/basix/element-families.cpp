// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "element-families.h"
#include <map>
#include <stdexcept>

using namespace basix;

//-----------------------------------------------------------------------------
element::family element::str_to_family(std::string name)
{
  static const std::map<std::string, element::family> name_to_family
      = {{"Custom element", element::family::custom},
         {"Lagrange", element::family::P},
         {"Discontinuous Lagrange", element::family::DP},
         {"DPC", element::family::DPC},
         {"Brezzi-Douglas-Marini", element::family::BDM},
         {"Raviart-Thomas", element::family::RT},
         {"Nedelec 1st kind H(curl)", element::family::N1E},
         {"Nedelec 2nd kind H(curl)", element::family::N2E},
         {"Regge", element::family::Regge},
         {"Crouzeix-Raviart", element::family::CR},
         {"Bubble", element::family::Bubble},
         {"Serendipity", element::family::Serendipity}};

  auto it = name_to_family.find(name);
  if (it == name_to_family.end())
    throw std::runtime_error("Can't find name " + name);

  return it->second;
}
//-----------------------------------------------------------------------------
const std::string& element::family_to_str(element::family family)
{
  static const std::map<element::family, std::string> name_to_family
      = {{element::family::custom, "Custom element"},
         {element::family::P, "Lagrange"},
         {element::family::DP, "Discontinuous Lagrange"},
         {element::family::DPC, "DPC"},
         {element::family::BDM, "Brezzi-Douglas-Marini"},
         {element::family::RT, "Raviart-Thomas"},
         {element::family::N1E, "Nedelec 1st kind H(curl)"},
         {element::family::N2E, "Nedelec 2nd kind H(curl)"},
         {element::family::Regge, "Regge"},
         {element::family::CR, "Crouzeix-Raviart"},
         {element::family::Bubble, "Bubble"},
         {element::family::Serendipity, "Serendipity"}};

  auto it = name_to_family.find(family);
  if (it == name_to_family.end())
    throw std::runtime_error("Can't find family");

  return it->second;
}
//-----------------------------------------------------------------------------
element::variant element::str_to_variant(std::string name)
{
  static const std::map<std::string, element::variant> name_to_variant
      = {{"default", element::variant::DEFAULT},
         {"equispaced", element::variant::EQ},
         {"Gauss-Lobatto-Legendre", element::variant::GLL}};

  auto it = name_to_variant.find(name);
  if (it == name_to_variant.end())
    throw std::runtime_error("Can't find name " + name);

  return it->second;
}
//-----------------------------------------------------------------------------
const std::string& element::variant_to_str(element::variant variant)
{
  static const std::map<element::variant, std::string> name_to_variant
      = {{element::variant::DEFAULT, "default"},
         {element::variant::EQ, "equispaced"},
         {element::variant::GLL, "Gauss-Lobatto-Legendre"}};

  auto it = name_to_variant.find(variant);
  if (it == name_to_variant.end())
    throw std::runtime_error("Can't find variant");

  return it->second;
}
//-----------------------------------------------------------------------------
lattice::type element::variant_to_lattice(element::variant variant)
{
  static const std::map<element::variant, lattice::type> name_to_variant
      = {{element::variant::EQ, lattice::type::equispaced},
         {element::variant::GLL, lattice::type::gll_warped}};

  auto it = name_to_variant.find(variant);
  if (it == name_to_variant.end())
    throw std::runtime_error("Can't find variant");

  return it->second;
}
//-----------------------------------------------------------------------------
element::variant element::lattice_to_variant(lattice::type lattice_type)
{
  static const std::map<lattice::type, element::variant> name_to_variant
      = {{lattice::type::equispaced, element::variant::EQ},
         {lattice::type::gll_warped, element::variant::GLL}};

  auto it = name_to_variant.find(lattice_type);
  if (it == name_to_variant.end())
    throw std::runtime_error("Can't find lattice type");

  return it->second;
}
//-----------------------------------------------------------------------------
