// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include <Eigen/Dense>
#include <vector>

#pragma once

namespace libtab
{

/// Information about reference cells
/// including their topological and geometric data.

namespace Cell
{

/// Cell type
enum class Type
{
  point,
  interval,
  triangle,
  tetrahedron,
  quadrilateral,
  hexahedron,
  prism,
  pyramid
};

/// Cell geometry
/// @param celltype Cell Type
/// @return Set of vertex points of the cell
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
geometry(Cell::Type celltype);

/// Cell topology
/// @param celltype Cell Type
/// @return List of topology (vertex indices) for each dimension (0..tdim)
std::vector<std::vector<std::vector<int>>> topology(Cell::Type celltype);

/// Sub-entity a cell, given by topological dimension and index
/// @param celltype The Cell::Type
/// @param dim Dimension of sub-entity
/// @param index Local index of sub-entity
/// @return Set of vertex points of the sub-entity
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
sub_entity_geometry(Cell::Type celltype, int dim, int index);

/// Number of sub-entities of a cell by topological dimension
/// @param celltype The Cell::Type
/// @param dim Dimension of sub-entity
/// @return The number of sub-entities of the given dimension
int sub_entity_count(Cell::Type celltype, int dim);

/// Create a lattice of points on the cell, equispaced along each edge
/// optionally including the outer surface points
/// @param celltype The Cell::Type
/// @param n number of points in each direction
/// @param exterior If set, includes outer boundaries
/// @return Set of points
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
create_lattice(Cell::Type celltype, int n, bool exterior);

/// Get the topological dimension for a given cell type
/// @param celltype Cell type
/// @return the topological dimension
int topological_dimension(Cell::Type celltype);

/// Get the cell type of a simplex of given dimension
Cell::Type simplex_type(int dim);

} // namespace Cell
} // namespace libtab
