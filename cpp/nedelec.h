// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include "finite-element.h"

#pragma once

class Nedelec2D : public FiniteElement
{
  /// Nedelec element (first kind) in 2D of order k
public:
  Nedelec2D(int k);
};

class Nedelec3D : public FiniteElement
{
  /// Nedelec element (first kind) in 3D of order k
public:
  Nedelec3D(int k);
};
