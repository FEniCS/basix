// FIXME: just include everything for now
// Need to define public API
// #include "cell.h"
// #include "crouzeix-raviart.h"
// #include "defines.h"
// #include "dof-permutations.h"
#include "finite-element.h"
// #include "indexing.h"
// #include "integral-moments.h"
// #include "lagrange.h"
// #include "lattice.h"
// #include "nedelec.h"
// #include "nedelec-second-kind.h"
// #include "polynomial-set.h"
// #include "quadrature.h"
// #include "raviart-thomas.h"
// #include "regge.h"
// #include "util.h"
#include <string>

namespace libtab
{
/// Create an element by name
FiniteElement create_element(std::string family, std::string cell, int degree);
} // namespace libtab
