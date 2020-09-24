// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "finite-element.h"

#pragma once

class Nedelec : public FiniteElement
{
  /// Nedelec element (first kind) of order k
public:
  /// Constructor
  /// @param celltype
  /// @param k degree
  Nedelec(Cell::Type celltype, int k);

  /// Tabulate basis at points
  /// @param pts Points
  /// @return Basis values at points
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  tabulate_basis(const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>& pts) const;
};
