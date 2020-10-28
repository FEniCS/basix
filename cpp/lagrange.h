// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "cell.h"
#include "finite-element.h"
#include <Eigen/Dense>
#include <vector>

#pragma once

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
};

class DiscontinuousLagrange
{
  /// Discontinuous Lagrange element
public:
  /// Discontinuous Lagrange element on cell with given degree
  /// @param celltype interval, triangle or tetrahedral celltype
  /// @param degree
  static FiniteElement create(cell::Type celltype, int degree);
};
} // namespace libtab
