// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "finite-element.h"

#pragma once

namespace libtab
{
class NedelecSecondKind
{
  /// Nedelec element (second kind) of order k
public:
  /// Constructor
  /// @param celltype
  /// @param k degree
  static FiniteElement create(cell::Type celltype, int k);
};
} // namespace libtab
