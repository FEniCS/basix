// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "finite-element.h"

#pragma once

namespace libtab
{
class CrouzeixRaviart
{
  /// CrouzeixRaviart element
public:
  /// Constructor
  /// @param celltype
  /// @param k degree - NB this must be 1 for Crouzeix-Raviart
  static FiniteElement create(cell::Type celltype, int k);
};
} // namespace libtab
