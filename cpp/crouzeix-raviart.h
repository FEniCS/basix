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
  /// @param celltype
  static FiniteElement create(cell::Type celltype);
};
} // namespace libtab
