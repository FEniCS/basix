// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "finite-element.h"

#pragma once

namespace libtab
{
class Regge : public FiniteElement
{
  /// Regge element of order k
public:
  /// Constructor
  /// @param celltype
  /// @param k degree
  Regge(Cell::Type celltype, int k);
};
} // namespace libtab
