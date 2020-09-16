// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "finite-element.h"

#pragma once

class RaviartThomas : public FiniteElement
{
  /// Raviart-Thomas element of given dimension (2 or 3) and degree k .
public:
  RaviartThomas(int dim, int k);
};
