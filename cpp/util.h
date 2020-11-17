
#pragma once

#include "finite-element.h"
#include<string>

namespace libtab
{
/// Create an element by name
FiniteElement create_element(std::string family, std::string cell, int degree);
}
