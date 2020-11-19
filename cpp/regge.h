// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "finite-element.h"

namespace libtab
{
class Regge
{
  /// Regge element
public:
  /// Regge element
  /// @param celltype
  /// @param degree
  static FiniteElement create(cell::Type celltype, int degree);

  inline static const std::string family_name = "Regge";
};
} // namespace libtab
