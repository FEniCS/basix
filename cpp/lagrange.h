// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include "finite-element.h"
#include <Eigen/Dense>
#include <vector>

namespace libtab
{
/// Lagrange element
namespace lagrange
{
/// Lagrange element on cell with given degree
/// @param celltype interval, triangle or tetrahedral celltype
/// @param degree
FiniteElement create(cell::Type celltype, int degree);

static std::string family_name = "Lagrange";
} // namespace lagrange

/// Discontinuous Lagrange element
namespace dlagrange
{
/// Discontinuous Lagrange element on cell with given degree
/// @param celltype interval, triangle or tetrahedral celltype
/// @param degree
FiniteElement create(cell::Type celltype, int degree);

static std::string family_name = "Discontinuous Lagrange";
} // namespace dlagrange
} // namespace libtab
