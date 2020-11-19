// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "finite-element.h"
#include <Eigen/Dense>
#include <vector>

namespace libtab
{
class Lagrange
{
  /// Lagrange element
public:
  /// Lagrange element on cell with given degree
  /// @param celltype interval, triangle or tetrahedral celltype
  /// @param degree
  static FiniteElement create(cell::Type celltype, int degree);

  inline static const std::string family_name = "Lagrange";
};

class DiscontinuousLagrange
{
  /// Discontinuous Lagrange element
public:
  /// Discontinuous Lagrange element on cell with given degree
  /// @param celltype interval, triangle or tetrahedral celltype
  /// @param degree
  static FiniteElement create(cell::Type celltype, int degree);

  inline static const std::string family_name = "Discontinuous Lagrange";
};
} // namespace libtab
