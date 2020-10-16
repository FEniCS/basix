// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "finite-element.h"

#pragma once

namespace libtab
{
class Regge : public FiniteElement
{
  /// Regge element of order k
public:
  /// Constructor
  /// @param celltype
  /// @param k degree
  Regge(Cell::Type celltype, int k);

  /// Tabulate basis and derivatives at points.
  /// The returned arrays have one row for each point, and the basis derivative
  /// values for each component are stacked in the columns, with XXXYYYZZZ
  /// ordering.
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
  tabulate(int nderiv,
           const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                              Eigen::RowMajor>& pts) const;
};
} // namespace libtab
