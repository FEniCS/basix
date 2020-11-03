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
class TensorProduct
{
public:
  /// Tensor Product element on cell with given degree
  /// @param celltype
  /// @param degree
  static FiniteElement create(cell::Type celltype, int degree);
};
} // namespace libtab
