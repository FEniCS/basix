// Copyright (c) 2020 Chris Richardson & Garth Wells
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include <vector>
#include <xtensor/xtensor.hpp>

namespace basix::lattice
{
/// The type of point spacing to be used in a lattice.
/// lattice::type::equispaced represents equally spaced points
/// on an interval and a regularly spaced set of points on other
/// shapes. lattice::type:gll represents the GLL (Gauss-Lobatto-Legendre)
/// points on an interval. Fot other shapes, the points used are obtained
/// by warping an equispaced grid of points, as described in Hesthaven and
/// Warburton, Nodal Discontinuous Galerkin Methods, 2008, pp 175-180
/// (https://doi.org/10.1007/978-0-387-72067-8).
enum class type
{
  equispaced = 0,
  gll = 1
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
/// "gll". The "gll" lattice has points spaced along each edge at
/// the Gauss-Lobatto-Legendre quadrature points. These are the same as
/// "equispaced" when n<3.
///
/// @param celltype The cell::type
/// @param n Size in each direction. There are n+1 points along each edge of the
/// cell.
/// @param type Either lattice::type::equispaced or lattice::type::gll
/// @param exterior If set, includes outer boundaries
/// @return Set of points
xt::xtensor<double, 2> create(cell::type celltype, int n, lattice::type type,
                              bool exterior);

} // namespace basix::lattice
