// Copyright (c) 2020 Chris Richardson & Matthew Scroggs
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "finite-element.h"

#pragma once

namespace libtab
{
class NedelecSecondKind : public FiniteElement
{
  /// Nedelec element (second kind) of order k
public:
  /// Constructor
  /// @param celltype
  /// @param k degree
  NedelecSecondKind(cell::Type celltype, int k);
};
} // namespace libtab
