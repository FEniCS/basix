// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "cell.h"
#include "finite-element.h"
#include <Eigen/Dense>
#include <vector>

#pragma once

class Lagrange : public FiniteElement
{
  /// Lagrange element
public:
  /// Constructor
  /// Lagrange element on cell with given degree
  /// @param celltype interval, triangle or tetrahedral celltype
  /// @param degree
  Lagrange(Cell::Type celltype, int degree);

  /// Tabulate basis at points
  /// The returned array has a row of basis values for each point
  /// @param pts Points
  /// @return Basis values at points
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  tabulate_basis(const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>& pts) const;

  /// Tabulate basis derivatives at points
  ///
  /// Each derivative up to the given order is returned, e.g. in 2D, for
  /// nderiv=2, 6 tables will be returned, for N, dN/dx, dN/dy, d2N/dx2,
  /// d2N/dxdy, d2N/dy2.
  ///
  /// @param pts Points
  /// @param nderiv Number of derivatives
  /// @return List of basis derivative values at points
  std::vector<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
  tabulate_basis_derivatives(
      int nderiv, const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                     Eigen::RowMajor>& pts) const;
};
