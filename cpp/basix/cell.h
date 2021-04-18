// Copyright (c) 2020 Chris Richardson
// FEniCS Project
// SPDX-License-Identifier:    MIT

#pragma once

#include <vector>
#include <xtensor/xtensor.hpp>

namespace basix
{

/// Information about reference cells including their topological and
/// geometric data.

namespace cell
{

/// Cell type
enum class type
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
xt::xtensor<double, 2> geometry(cell::type celltype);

/// Cell topology
/// @param celltype Cell Type
/// @return List of topology (vertex indices) for each dimension (0..tdim)
std::vector<std::vector<std::vector<int>>> topology(cell::type celltype);

/// Sub-entity of a cell, given by topological dimension and index
/// @param celltype The cell::type
/// @param dim Dimension of sub-entity
/// @param index Local index of sub-entity
/// @return Set of vertex points of the sub-entity
xt::xtensor<double, 2> sub_entity_geometry(cell::type celltype, int dim,
                                           int index);

/// @todo Optimise this function
/// Number of sub-entities of a cell by topological dimension
/// @param celltype The cell::type
/// @param dim Dimension of sub-entity
/// @return The number of sub-entities of the given dimension
/// @warning This function is expensive to call. Do not use in
/// performance critical code
int num_sub_entities(cell::type celltype, int dim);

/// Get the topological dimension for a given cell type
/// @param celltype Cell type
/// @return the topological dimension
int topological_dimension(cell::type celltype);

/// Get the cell type of a sub-entity of given dimension and index
/// @param celltype Type of cell
/// @param dim Topological dimension of sub-entity
/// @param index Index of sub-entity
/// @return cell type of sub-entity
cell::type sub_entity_type(cell::type celltype, int dim, int index);

/// Convert a cell type string to enum
/// @param name String
/// @return cell type
cell::type str_to_type(std::string name);

/// Convert cell type enum to string
const std::string& type_to_str(cell::type type);

} // namespace cell
} // namespace basix
