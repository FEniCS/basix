#include "util.h"

#include "crouzeix-raviart.h"
#include "lagrange.h"
#include "nedelec-second-kind.h"
#include "nedelec.h"
#include "raviart-thomas.h"
#include "regge.h"

#include <map>

using namespace libtab;

FiniteElement libtab::create_element(std::string family, std::string cell,
                                     int degree)
{
  const std::map<std::string, std::function<FiniteElement(cell::Type, int)>>
      create_map
      = {{CrouzeixRaviart::family_name, &CrouzeixRaviart::create},
         {DiscontinuousLagrange::family_name, &DiscontinuousLagrange::create},
         {Lagrange::family_name, &Lagrange::create},
         {Nedelec::family_name, &Nedelec::create},
         {NedelecSecondKind::family_name, &NedelecSecondKind::create},
         {RaviartThomas::family_name, &RaviartThomas::create},
         {Regge::family_name, &Regge::create}};

  auto create_it = create_map.find(family);
  if (create_it == create_map.end())
    throw std::runtime_error("Family not found: \"" + family + "\"");

  return create_it->second(cell::str_to_type(cell), degree);
}
