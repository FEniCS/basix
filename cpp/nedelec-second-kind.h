// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "finite-element.h"

namespace libtab
{
class NedelecSecondKind
{
  /// Nedelec element (second kind)
public:
  /// @param celltype
  /// @param degree
  static FiniteElement create(cell::Type celltype, int degree);

  inline static const std::string family_name = "Nedelec 2nd kind H(curl)";
};
} // namespace libtab
