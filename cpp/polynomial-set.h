// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "cell.h"
#include "polynomial.h"
#include <vector>

#pragma once

class PolynomialSet
{
public:
  /// Orthonormal polynomial basis on reference cell
  /// @param celltype Cell type
  /// @param n order
  /// @return polynomial set
  static std::vector<Polynomial> compute_polynomial_set(Cell::Type celltype,
                                                        int n);

  static Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  tabulate_polynomial_set(
      Cell::Type celltype, int n,
      const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                         Eigen::RowMajor>& pts);
};
