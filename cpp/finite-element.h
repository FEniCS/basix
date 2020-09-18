// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "cell.h"
#include "polynomial.h"
#include <Eigen/Dense>
#include <vector>

#pragma once

class FiniteElement
{
  /// Finite element base class, taking the spatial dimension and degree,
  /// storing basis as a polynomial set.

public:
  /// Element of given dimension (1, 2 or 3) and degree.
  FiniteElement(Cell::Type cell_type, int degree);

  /// Compute basis values at set of points. If a vector result is expected, it
  /// will be stacked with all x values, followed by all y-values (and then z,
  /// if any).
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  tabulate_basis(const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>& pts) const;

protected:
  // Applies nodal constraints from dualmat to original coeffs on basis, and
  // stores to polynomial set.
  void apply_dualmat_to_basis(
      const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::RowMajor>& coeffs,
      const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                          Eigen::RowMajor>& dualmat,
      const std::vector<Polynomial>& basis, int ndim);

  // cell type
  Cell::Type _cell_type;

  // degree
  int _degree;

  // set of polynomials
  std::vector<Polynomial> poly_set;
};
