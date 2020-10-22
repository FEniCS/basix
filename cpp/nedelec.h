// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "finite-element.h"

#pragma once

namespace libtab
{
class Nedelec : public FiniteElement
{
  /// Nedelec element (first kind) of order k
public:
  /// Constructor
  /// @param celltype
  /// @param k degree
  Nedelec(Cell::Type celltype, int k);
};
} // namespace libtab
