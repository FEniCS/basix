// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "polynomial.h"
#include <Eigen/Dense>
#include <vector>

#pragma once

class Nedelec2D
{
public:
  Nedelec2D(int k);

private:
  int _degree;
  std::vector<Polynomial> poly_set;
};
