// FIXME: just include everything for now
// Need to define public API

#include "finite-element.h"
#include <string>

namespace libtab
{
/// Create an element by name
FiniteElement create_element(std::string family, std::string cell, int degree);
} // namespace libtab
