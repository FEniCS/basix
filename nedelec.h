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

  // Compute basis values at set of points
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  tabulate_basis(const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>& pts) const;

private:
  int _dim;
  int _degree;
  std::vector<Polynomial> poly_set;
};
