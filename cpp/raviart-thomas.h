// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "finite-element.h"

namespace libtab
{
class RaviartThomas
{
  /// Raviart-Thomas element
public:
  /// @param celltype
  /// @param degree
  static FiniteElement create(cell::Type celltype, int degree);

  inline static const std::string family_name = "Raviart-Thomas";
};
} // namespace libtab
