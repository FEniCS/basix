// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "cell.h"

#include <vector>

#pragma once


// Compute indexing in a 2D triangular array compressed into a 1D array
static inline int idx(int p, int q) { return (p + q + 1) * (p + q) / 2 + q; }

// Compute indexing in a 3D tetrahedral array compressed into a 1D array
static inline int idx(int p, int q, int r)
{
  return (p + q + r) * (p + q + r + 1) * (p + q + r + 2) / 6
         + (q + r) * (q + r + 1) / 2 + r;
};

namespace PolynomialSet
{
  /// Orthonormal polynomial basis on reference cell
  /// @param celltype Cell type
  /// @param n order
  /// @param pts points
  /// @return Polynomial set tabulated at points
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  tabulate_polynomial_set(
      Cell::Type celltype, int n,
      const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                         Eigen::RowMajor>& pts);

  /// Derivative of orthonormal polynomial basis on reference cell
  /// TODO
  /// @param celltype Cell type
  /// @param n order
  /// @param nderivs Maximum derivative order
  /// @param pts points
  /// @return List of polynomial sets, for each derivative.
  /// tabulated at points. List ordering is triangular using idx()
  std::vector<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
  tabulate_polynomial_set_deriv(
                                Cell::Type celltype, int n, int nderiv,
      const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                         Eigen::RowMajor>& pts);
};
