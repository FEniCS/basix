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
class Lagrange : public FiniteElement
{
  /// Lagrange element
public:
  /// Constructor
  /// Lagrange element on cell with given degree
  /// @param celltype interval, triangle or tetrahedral celltype
  /// @param degree
  Lagrange(cell::Type celltype, int degree);
};

class DiscontinuousLagrange : public FiniteElement
{
  /// Discontinuous Lagrange element
public:
  /// Constructor
  /// Discontinuous Lagrange element on cell with given degree
  /// @param celltype interval, triangle or tetrahedral celltype
  /// @param degree
  DiscontinuousLagrange(cell::Type celltype, int degree);
};
} // namespace libtab
