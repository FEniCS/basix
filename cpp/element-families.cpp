// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "element-families.h"
#include <stdexcept>

using namespace basix;

//-----------------------------------------------------------------------------
element::family element::str_to_family(std::string family)
{
  if (family == "Lagrange" or family == "P" or family == "Q")
    return element::family::P;
  else if (family == "Discontinuous Lagrange")
    return element::family::DP;
  else if (family == "Brezzi-Douglas-Marini")
    return element::family::BDM;
  else if (family == "Raviart-Thomas")
    return element::family::RT;
  else if (family == "Nedelec 1st kind H(curl)")
    return element::family::N1E;
  else if (family == "Nedelec 2nd kind H(curl)")
    return element::family::N2E;
  else if (family == "Regge")
    return element::family::Regge;
  else if (family == "Crouzeix-Raviart")
    return element::family::CR;
  else
    throw std::runtime_error("Family not found: \"" + family + "\"");
}
//-----------------------------------------------------------------------------
std::string element::family_to_str(element::family family)
{
  if (family == element::family::P)
    return "Lagrange";
  else if (family == element::family::DP)
    return "Discontinuous Lagrange";
  else if (family == element::family::BDM)
    return "BDM";
  else if (family == element::family::RT)
    return "RT";
  else if (family == element::family::N1E)
    return "N1E";
  else if (family == element::family::N2E)
    return "N2E";
  else if (family == element::family::Regge)
    return "Regge";
  else if (family == element::family::CR)
    return "Crouzeix-Raviart";
  else
    throw std::runtime_error("Family not found");
}
//-----------------------------------------------------------------------------
