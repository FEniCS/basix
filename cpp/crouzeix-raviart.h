// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "finite-element.h"

namespace libtab
{
class CrouzeixRaviart
{
  /// Crouzeix-Raviart element
public:
  /// @note degree must be 1 for Crouzeix-Raviart
  /// @param celltype
  /// @param degree
  static FiniteElement create(cell::Type celltype, int degree);
};
} // namespace libtab
