// Copyright (c) 2020 Chris Richardson & Garth Wells
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include <xtensor/xtensor.hpp>

/// Lattices of points
namespace basix::lattice
{
/// The type of point spacing to be used in a lattice.
///
/// lattice::type::equispaced represents equally spaced points
/// on an interval and a regularly spaced set of points on other
/// shapes.
///
/// lattice::type::gll represents the GLL (Gauss-Lobatto-Legendre)
/// points.
///
/// lattice::type::chebyshev represents the Chebyshev points.
///
/// lattice::type::gl represents the GL (Gauss-Legendre) points.
///
/// lattice::type::chebyshev_plus_endpoints represents the Chebyshev points plus
/// the endpoints of the interval. lattice::type::gl_plus_endpoints represents
/// the Chebyshev points plus the endpoints of the interval. These points are
/// only intended for internal use.
enum class type
{
  equispaced = 0,
  gll = 1,
  chebyshev = 2,
  gl = 4,

  chebyshev_plus_endpoints = 10,
  gl_plus_endpoints = 11,
};

/// The method used to generate points inside simplices.
///
/// lattice::simplex_method::none can be used when no method is needed (eg when
/// making points on a quadrilateral, or when making equispaced points).
///
/// lattice::simplex_method::warp will use the warping defined in Hesthaven and
/// Warburton, Nodal Discontinuous Galerkin Methods, 2008, pp 175-180
/// (https://doi.org/10.1007/978-0-387-72067-8).
///
/// lattice::simplex_method::isaac will use the method described in Isaac,
/// Recursive, Parameter-Free, Explicitly Defined Interpolation Nodes for
/// Simplices, 2020 (https://doi.org/10.1137/20M1321802).
///
/// lattice::simplex_method::centroid will place points at the centroids of the
/// grid created by putting points on the edges, as described in Blyth and
/// Pozrikidis, A Lobatto interpolation grid over the triangle, 2001
/// (https://doi.org/10.1093/imamat/hxh077).
enum class simplex_method
{
  none = 0,
  warp = 1,
  isaac = 2,
  centroid = 3,
};

/// Create a lattice of points on a reference cell
/// optionally including the outer surface points
///
/// For a given celltype, this creates a set of points on a regular grid
/// which covers the cell, eg for a quadrilateral, with n=2, the points are:
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
/// @param type A lattice type
/// @param exterior If set, includes outer boundaries
/// @param simplex_method The method used to generate points on simplices
/// @return Set of points
xt::xtensor<double, 2>
create(cell::type celltype, int n, lattice::type type, bool exterior,
       lattice::simplex_method simplex_method = lattice::simplex_method::none);

} // namespace basix::lattice
