// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "cell.h"
#include "finite-element.h"
#include <Eigen/Dense>
#include <vector>

#pragma once

class TensorProduct : public FiniteElement
{
public:
  /// Tensor Product element on cell with given degree
  /// @param celltype
  /// @param degree
  TensorProduct(Cell::Type celltype, int degree);

  /// Tabulate basis at points
  /// @param pts Points
  /// @return Basis values at points
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  tabulate_basis(const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>& pts) const;
};
