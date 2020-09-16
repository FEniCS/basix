// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "finite-element.h"

#pragma once

class Nedelec2D : public FiniteElement
{
public:
  Nedelec2D(int k);
};

class Nedelec3D : public FiniteElement
{
public:
  Nedelec3D(int k);
};
