// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "finite-element.h"

namespace libtab
{
class Nedelec
{
  /// Nedelec element (first kind)
public:
  /// @param celltype
  /// @param degree
  static FiniteElement create(cell::Type celltype, int degree);

  inline static const std::string family_name = "Nedelec 1st kind H(curl)";
};
} // namespace libtab
