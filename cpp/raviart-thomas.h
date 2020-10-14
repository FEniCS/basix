// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "cell.h"
#include "finite-element.h"

#pragma once

namespace libtab
{
class RaviartThomas : public FiniteElement
{
  /// Raviart-Thomas element of order k
public:
  /// Constructor
  /// @param celltype
  /// @param k degree
  RaviartThomas(Cell::Type celltype, int k);

  /// Tabulate basis and derivatives at points
  /// @param nderiv Derivative order
  /// @param pts Points
  /// @return Basis values at points
  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
  tabulate(int nderiv,
           const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                              Eigen::RowMajor>& pts) const;
};
} // namespace libtab
