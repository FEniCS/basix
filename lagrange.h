// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "finite-element.h"
#include "polynomial.h"
#include <Eigen/Dense>
#include <vector>

#pragma once

class Lagrange : public FiniteElement
{
  /// Lagrange element of given dimension (1, 2 or 3) and degree.
public:
  Lagrange(int dim, int degree);
};
