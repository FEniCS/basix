// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "cell.h"
#include <Eigen/Dense>
#include <vector>

#pragma once

class FiniteElement
{
  /// Finite element base class, taking the cell type and degree,
  /// The basis is stored as a set of coefficients, which are applied to the
  /// underlying expansion set for that cell type, when tabulating.

public:
  /// Element of given dimension (1, 2 or 3) and degree.
  FiniteElement(Cell::Type cell_type, int degree);

  /// Compute basis values at set of points. If a vector result is expected, it
  /// will be stacked with all x values, followed by all y-values (and then z,
  /// if any).
  virtual Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  tabulate_basis(const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>& pts) const = 0;

protected:
  // Applies nodal constraints from dualmat to original coeffs on basis, and
  // stores to polynomial set.
  void apply_dualmat_to_basis(
      const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::RowMajor>& coeffs,
      const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
      Eigen::RowMajor>& dualmat);


  // cell type
  Cell::Type _cell_type;

  // degree
  int _degree;

  // Coefficient of expansion sets on cell
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    _coeffs;

};
