// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "finite-element.h"
#include "polynomial.h"
#include "cell.h"
#include <Eigen/Dense>
#include <vector>

#pragma once

class TensorProduct : public FiniteElement
{
public:
  TensorProduct(Cell::Type celltype, int degree);
};
