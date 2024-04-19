// Copyright (c) 2020-2022 Chris Richardson and Garth N. Wells
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include "cell.h"
#include <array>
#include <concepts>
#include <utility>
#include <vector>

/// @brief Lattices of points
namespace basix::lattice
{
/// @brief The type of point spacing to be used in a lattice.
///
/// @note type::chebyshev_plus_endpoints() and type::gl_plus_endpoints() are
/// only intended for internal use only.
enum class type
{
  equispaced = 0, /*!< Equally spaced points */
  gll = 1,        /*!< Gauss-Lobatto-Legendre (GLL) points */
  chebyshev = 2,  /*!< Chebyshev points */
  gl = 4,         /*!< Gauss-Legendre (GL) points */
  chebyshev_plus_endpoints
  = 10, /*!< Chebyshev points plus the endpoints of the interval */
  gl_plus_endpoints
  = 11, /*!< Gauss-Legendre (GL) points plus the endpoints of the interval */
};

/// @brief The method used to generate points inside simplices.
enum class simplex_method
{
  none = 0,     /*!< Used when no method is needed, e.g. when making points on a
                   quadrilateral, or when making equispaced points). */
  warp = 1,     /*!< Warping from Hesthaven and Warburton, Nodal Discontinuous
                   Galerkin Methods, https://doi.org/10.1007/978-0-387-72067-8. */
  isaac = 2,    /*!<  Points described in Isaac, Recursive, Parameter-Free,
                   Explicitly Defined Interpolation Nodes for Simplices,
                   https://doi.org/10.1137/20M1321802. */
  centroid = 3, /*!< Place points at the centroids of the grid created by
                   putting points on the edges, as described in Blyth and
                   Pozrikidis, A Lobatto interpolation grid over the triangle,
                   https://doi.org/10.1093/imamat/hxh077. */
};

/// @brief Create a lattice of points on a reference cell optionally
/// including the outer surface points.
///
/// For a given `celltype`, this creates a set of points on a regular
/// grid which covers the cell, eg for a quadrilateral, with n=2, the
/// points are: `[0,0], [0.5,0], [1,0], [0,0.5], [0.5,0.5], [1,0.5],
/// [0,1], [0.5,1], [1,1]`. If the parameter exterior is set to false,
/// the points lying on the external boundary are omitted, in this case
/// for a quadrilateral with `n == 2`, the points are: `[0.5, 0.5]`. The
/// lattice type can be chosen as type::equispaced or type::gll. The
/// type::gll lattice has points spaced along each edge at the
/// Gauss-Lobatto-Legendre quadrature points. These are the same as
/// type::equispaced when `n < 3`.
///
/// @param celltype The cell type.
/// @param n Size in each direction. There are `n + 1` points along each
/// edge of the cell.
/// @param type A lattice type.
/// @param exterior If set, includes outer boundaries.
/// @param simplex_method The method used to generate points on
/// simplices.
/// @return Set of points. Shape is `(npoints, tdim)` and storage is
/// row-major.
template <std::floating_point T>
std::pair<std::vector<T>, std::array<std::size_t, 2>>
create(cell::type celltype, int n, lattice::type type, bool exterior,
       lattice::simplex_method simplex_method = lattice::simplex_method::none);

} // namespace basix::lattice
