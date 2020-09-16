// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "finite-element.h"

#pragma once

class RaviartThomas : public FiniteElement
{
public:
  RaviartThomas(int dim, int k);
};
