// Copyright (c) 2020 Chris Richardson & Garth Wells
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include <vector>
#include <xtensor/xtensor.hpp>

namespace basix::lattice
{
enum class type
{
  undefined,
  equispaced,
  gll_warped
};

/// Create a lattice of points on a reference cell
/// optionally including the outer surface points
///
/// For a given celltype, this creates a set of points on a regular grid
/// which covers the cell, e.g. for a quadrilateral, with n=2, the points are:
/// [0,0],[0.5,0],[1,0],[0,0.5],[0.5,0.5],[1,0.5],[0,1],[0.5,1],[1,1]
/// If the parameter exterior is set to false, the points lying on the external
/// boundary are omitted, in this case for a quadrilateral with n=2, the points
/// are: [0.5,0.5]. The lattice type can be chosen as "equispaced" or
/// "gll_warped". The "gll_warped" lattice has points spaced along each edge at
/// the Gauss-Lobatto-Legendre quadrature points. These are the same as
/// "equispaced" when n<3.
///
/// @param celltype The cell::type
/// @param n Size in each direction. There are n+1 points along each edge of the
/// cell.
/// @param type Either lattice::type::equispaced or lattice::type::gll_warped
/// @param exterior If set, includes outer boundaries
/// @return Set of points
xt::xtensor<double, 2> create(cell::type celltype, int n, lattice::type type,
                              bool exterior);

} // namespace basix::lattice
