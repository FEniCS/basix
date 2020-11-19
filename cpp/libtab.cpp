// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "libtab.h"
#include "crouzeix-raviart.h"
#include "finite-element.h"
#include "lagrange.h"
#include "nedelec.h"
#include "polynomial-set.h"
#include "raviart-thomas.h"
#include "regge.h"
#include <map>

using namespace libtab;

//-----------------------------------------------------------------------------
libtab::FiniteElement libtab::create_element(std::string family,
                                             std::string cell, int degree)
{
  const std::map<std::string, std::function<FiniteElement(cell::Type, int)>>
      create_map = {{cr::family_name, &cr::create},
                    {dlagrange::family_name, &dlagrange::create},
                    {lagrange::family_name, &lagrange::create},
                    {nedelec::family_name, &nedelec::create},
                    {nedelec2::family_name, &nedelec2::create},
                    {rt::family_name, &rt::create},
                    {regge::family_name, &regge::create}};

  auto create_it = create_map.find(family);
  if (create_it == create_map.end())
    throw std::runtime_error("Family not found: \"" + family + "\"");

  return create_it->second(cell::str_to_type(cell), degree);
}
//-----------------------------------------------------------------------------
