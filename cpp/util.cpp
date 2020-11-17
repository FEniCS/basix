#include "util.h"

#include "crouzeix-raviart.h"
#include "lagrange.h"
#include "nedelec-second-kind.h"
#include "nedelec.h"
#include "raviart-thomas.h"
#include "regge.h"

#include <map>

using namespace libtab;

FiniteElement create_element(std::string family, std::string cell, int degree)
{
  const std::map<std::string, std::function<FiniteElement(cell::Type, int)>>
      create_map = {{"Crouzeix-Raviart", &CrouzeixRaviart::create},
                    {"Discontinuous Lagrange", &Lagrange::create},
                    {"Lagrange", &Lagrange::create},
                    {"Nedelec 1st kind H(curl)", &Nedelec::create},
                    {"Nedelec 2nd kind H(curl)", &NedelecSecondKind::create},
                    {"Raviart-Thomas", &RaviartThomas::create},
                    {"Regge", &Regge::create}};

  auto create_it = create_map.find(family);
  if (create_it == create_map.end())
    throw std::runtime_error("Family not found: \"" + family + "\"");

  return create_it->second(cell::str_to_type(cell), degree);
}
