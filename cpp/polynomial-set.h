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
  static std::vector<Polynomial> compute_polynomial_set(CellType celltype, int n);

  static std::vector<Polynomial> compute_polyset_quad(int n);
};
