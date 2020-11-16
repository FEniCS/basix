// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#include <Eigen/Dense>
#include <vector>

#pragma once

namespace libtab
{

/// Information about reference cells including their topological and
/// geometric data.

namespace cell
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
geometry(cell::Type celltype);

/// Cell topology
/// @param celltype Cell Type
/// @return List of topology (vertex indices) for each dimension (0..tdim)
std::vector<std::vector<std::vector<int>>> topology(cell::Type celltype);

/// Sub-entity of a cell, given by topological dimension and index
/// @param celltype The cell::Type
/// @param dim Dimension of sub-entity
/// @param index Local index of sub-entity
/// @return Set of vertex points of the sub-entity
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
sub_entity_geometry(cell::Type celltype, int dim, int index);

/// Number of sub-entities of a cell by topological dimension
/// @param celltype The cell::Type
/// @param dim Dimension of sub-entity
/// @return The number of sub-entities of the given dimension
int sub_entity_count(cell::Type celltype, int dim);

/// Get the topological dimension for a given cell type
/// @param celltype Cell type
/// @return the topological dimension
int topological_dimension(cell::Type celltype);

/// Get the cell type of a sub-entity of given dimension and index
/// @param celltype Type of cell
/// @param dim Topological dimension of sub-entity
/// @param index Index of sub-entity
/// @return cell type of sub-entity
cell::Type sub_entity_type(cell::Type celltype, int dim, int index);

/// Convert a cell type string to enum
/// @param name String
/// @return cell type
cell::Type str_to_type(std::string name);

} // namespace cell
} // namespace libtab
