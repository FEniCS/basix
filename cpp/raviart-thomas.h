// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "cell.h"
#include "finite-element.h"

#pragma once

class RaviartThomas : public FiniteElement
{
  /// Raviart-Thomas element of order k
public:
  /// Constructor
  /// @param celltype
  /// @param k degree
  RaviartThomas(Cell::Type celltype, int k);

  /// Tabulate basis at points
  /// @param pts Points
  /// @return Basis values at points
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  tabulate_basis(const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>& pts) const;

  /// Tabulate basis derivatives at points
  /// @param nderiv Derivative order
  /// @param pts Points
  /// @return Basis values at points
  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
  tabulate_basis_derivatives(
      int nderiv, const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>& pts) const;
};
