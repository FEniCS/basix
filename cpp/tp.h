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
class TensorProduct : public FiniteElement
{
public:
  /// Tensor Product element on cell with given degree
  /// @param celltype
  /// @param degree
  TensorProduct(Cell::Type celltype, int degree);

  /// Tabulate basis and derivatives at points
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
